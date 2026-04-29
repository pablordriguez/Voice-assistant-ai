import vosk
import pyaudio
import json
import os
import re
import tempfile
import logging

# if you don't want to download the model, just mention "lang" argument
# in vosk.Model() and it will download the right  model, here the language is
# US-English
# model = vosk.Model(lang="en-us")

# Import the required module for text
# to speech conversion
from gtts import gTTS

# Import pygame for playing the converted audio
import pygame

import requests

from transformers import pipeline


# Basic logging for debugging and traceability
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B")
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe(messages)


# Normalize recognized text to reduce problems caused by punctuation/noise
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tell_something(mytext):
    # Language in which you want to convert
    language = 'en'

    try:
        # Passing the text and language to the engine,
        # here we have marked slow=False. Which tells
        # the module that the converted audio should
        # have a high speed
        myobj = gTTS(text=mytext, lang=language, slow=False)

        # Use temporary file instead of fixed "welcome.mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            filename = tmp.name

        # Saving the converted audio in a mp3 file named
        # welcome
        myobj.save(filename)

        # Initialize the mixer module
        pygame.mixer.init()

        # Load the mp3 file
        pygame.mixer.music.load(filename)

        # Play the loaded mp3 file
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        logger.error(f"TTS error: {e}")

    finally:
        # Delete temp file after playback
        try:
            if 'filename' in locals() and os.path.exists(filename):
                os.remove(filename)
        except Exception:
            pass


def ask_llm(text):
    conf_prompt = "The input text is a transcription, refine the prompt. Then apply the refined prompt as a prompt. The input text: "
    messages = [
        {"role": "user", "content": conf_prompt + " " + text},
    ]
    try:
        return pipe(
            messages,
            return_full_text=False,
            do_sample=True,
            max_new_tokens=512,   # Reduced from 1024 for faster responses
            num_workers=1         # Safer default
        )
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return [{"generated_text": "Sorry, I could not process that request."}]


def compare_strings_misspelled(given_string, wrong_string):
    # Convert two strings to lowercase to ignore case
    given_string = given_string.lower()
    wrong_string = wrong_string.lower()

    # Initialize a counter to keep track of the differences
    differences = 0

    if len(given_string) > len(wrong_string):
        tmp = wrong_string
        wrong_string = given_string
        given_string = tmp

    # Iterate over each character in the given string
    for charac in range(len(given_string)):
        # If character is in wrong string and the ASCII value of character in given string is larger than ASCII value of character in wrong string, increment differences
        if given_string[charac] in wrong_string and ord(given_string[charac]) > ord(wrong_string[charac]):
            differences += 1

    differences += (len(wrong_string) - len(given_string))

    # Return the total differences
    return differences


def find_word_in_list(word, lista):
    return sum([compare_strings_misspelled(word, s) < 2 for s in lista if s]) > 0


if __name__ == '__main__':

    # Here I have downloaded this model to my PC, extracted the files
    # and saved it in local directory
    # Set the model path
    model_path = "vosk-model-en-us-0.42-gigaspeech"

    # Initialize the model with model-path
    model = vosk.Model(model_path)

    # Use environment variables instead of hardcoding tokens
    tg_api_token = os.getenv("TG_API_TOKEN")
    tg_chat_id = os.getenv("TG_CHAT_ID")

    def send_tg_message(text='Cell execution completed.'):
        if not tg_api_token or not tg_chat_id:
            logger.warning("Telegram token/chat_id not configured")
            return

        try:
            requests.post(
                'https://api.telegram.org/' +
                'bot{}/sendMessage'.format(tg_api_token),
                params=dict(chat_id=tg_chat_id, text=text),
                timeout=10
            ).raise_for_status()
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    # Create a recognizer
    rec = vosk.KaldiRecognizer(model, 16000)

    # Open the microphone stream
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=8192
    )

    # Open a text file in write mode using a 'with' block
    # with open(output_file_path, "w") as output_file:
    print("Listening for speech. Say 'Terminate' to stop.")

    # Start streaming and recognize speech
    buffer = ""
    record = False

    try:
        while True:
            data = stream.read(1024, exception_on_overflow=False)  # read in chunks
            if rec.AcceptWaveform(data):  # accept waveform of input voice
                # Parse the JSON result and get the recognized text
                result = json.loads(rec.Result())
                recognized_text = result.get('text', '')
                recognized_text = normalize_text(recognized_text)

                # Write recognized text to the file
                # output_file.write(recognized_text + "\n")
                print("Recognized:", recognized_text)

                if not recognized_text:
                    continue

                lista = recognized_text.split(" ")

                if record and find_word_in_list("stop", lista):
                    print("-- Stop --")
                    tell_something("Codi stopped")
                    record = False
                    buffer = ""

                elif record and find_word_in_list("upload", lista):
                    tell_something("Codi upload")
                    print("-- UPLOAD --")
                    record = False

                    # Keep anything said before the upload keyword
                    upload_parts = recognized_text.split("upload")
                    buffer += "".join(upload_parts[:1]).strip()

                    if not buffer.strip():
                        tell_something("No content to upload")
                        continue

                    send_tg_message("Codi thinking about: " + buffer)
                    tell_something("You asked: " + buffer)

                    result = ask_llm(buffer)
                    answer = result[0]["generated_text"]

                    send_tg_message("Result:\n" + answer)
                    buffer = ""
                    print(answer)

                elif record:
                    # Add space between chunks to avoid joined words
                    buffer += " " + recognized_text.lower()

                elif not record and find_word_in_list("start", lista):
                    print("-- Start --")
                    tell_something("Codi started")
                    record = True

                    # Keep text after "start" if present
                    start_parts = recognized_text.split("start")
                    if len(start_parts) > 1:
                        buffer += " ".join(start_parts[1:]).strip()

                # Check for the termination keyword
                if find_word_in_list("terminate", lista):
                    tell_something("Codi terminate")
                    print("Termination keyword detected. Stopping...")
                    break

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        # Clean shutdown
        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception:
            pass

        try:
            pygame.mixer.quit()
        except Exception:
            pass