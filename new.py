import pyaudio
import torch
import numpy as np
from faster_whisper import WhisperModel
import os
import time
import json
from vosk import Model, KaldiRecognizer
import pyttsx3 

import ollama

# --- ------------------------------------------------------------------- --- #
# ---                         MAIN CONFIGURATION PANEL                    --- #
# --- ------------------------------------------------------------------- --- #

# --- VOSK Wake Word & End Call Configuration ---
WAKE_WORD = "hey there"
END_CALL_PHRASE = "kill this"
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

# --- Models and Devices ---
STT_MODEL_SIZE = "tiny.en" 
OLLAMA_MODEL = "gemma3:1b" 
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# --- Conversation Behavior Control ---
SILENCE_BEFORE_PROCESSING_SECONDS = 2.0 

# --- LLM Personality and Behavior Control ---
SYSTEM_PROMPT = "you are a sales agent working for company named Airtravellers your job is to call people and ask them if they are planning to travel and looking for best deals on all domestic and international flight, you have to be persuasuve, but you can never say anything about any offers or pricing or availablity of flights if asked such questions kindly ask the user to visit the website for latest information try to be super softspoken you can help or provide people with suggestions for places to visit to. "
OLLAMA_OPTIONS = {'temperature': 0.7, 'num_predict': 96, 'top_p': 0.9}


# --- NEW: VAD Tuning and Debugging ---
# The VAD sensitivity threshold. Lower values are more sensitive (more likely to detect speech).
# Start with 0.35 if your voice is not being picked up.
VAD_THRESHOLD = 0.35

# Set this to True to see the real-time VAD confidence score.
# Use this to find the perfect VAD_THRESHOLD for your microphone.
VAD_DEBUG_MODE = True

# --- ------------------------------------------------------------------- --- #
# ---                    END OF CONFIGURATION PANEL                       --- #
# --- ------------------------------------------------------------------- --- #


# --- Audio Settings & Initializations ---
INPUT_RATE = 16000
CHUNK_SIZE = 512 
VAD_MODEL, VAD_UTILS = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
(_, _, _, VADIterator, _) = VAD_UTILS
STATE_WAITING_FOR_WAKE_WORD = 1
STATE_LISTENING_FOR_COMMAND = 2

print("--- Initializing Models (this may take a moment) ---")

# 1. Initialize Vosk Wake Word Engine
if not os.path.exists(VOSK_MODEL_PATH):
    print(f"Vosk model not found at '{VOSK_MODEL_PATH}'.")
    exit()
print("Loading Vosk model...")
vosk_grammar = [WAKE_WORD, END_CALL_PHRASE, "[unk]"]
recognizer = KaldiRecognizer(Model(VOSK_MODEL_PATH), INPUT_RATE, json.dumps(vosk_grammar))

# 2. Initialize STT (faster-whisper)
print(f"Loading Whisper model: {STT_MODEL_SIZE}...")
stt_model = WhisperModel(STT_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

# 3. Initialize pyttsx3 TTS Engine
print("Loading System TTS Engine...")
tts_engine = pyttsx3.init()

# 4. --- FIX: Create VAD instance with the new configurable threshold ---
vad_iterator = VADIterator(VAD_MODEL, threshold=VAD_THRESHOLD)

print("--- All models initialized ---")


def speak_text_system(text, engine):
    print(f"Assistant: {text}")
    try:
        if engine.isBusy(): engine.stop()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error during TTS synthesis or playback: {e}")

def play_activation_sound(frequency=880):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=22050, output=True)
    duration = 0.1; volume = 0.3
    samples = (np.sin(2 * np.pi * np.arange(22050 * duration) * frequency / 22050)).astype(np.float32)
    stream.write(volume * samples)
    stream.close()
    p.terminate()

def ask_ollama(prompt, history, options):
    print("\n> Asking Ollama...")
    history.append({'role': 'user', 'content': prompt})
    response = ollama.chat(model=OLLAMA_MODEL, messages=history, stream=False, options=options)
    assistant_response = response['message']['content']
    history.append({'role': 'assistant', 'content': assistant_response})
    return assistant_response, history

def main():
    p_mic = pyaudio.PyAudio()
    mic_stream = p_mic.open(rate=INPUT_RATE, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=CHUNK_SIZE)
    
    current_state = STATE_WAITING_FOR_WAKE_WORD
    conversation_history = []
    
    voiced_frames = []
    silence_start_time = None
    user_is_speaking = False

    print(f"\n--- Ready! Listening for '{WAKE_WORD}' ---")

    try:
        while True:
            audio_chunk_raw = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            if current_state == STATE_WAITING_FOR_WAKE_WORD:
                if recognizer.AcceptWaveform(audio_chunk_raw):
                    result_json = json.loads(recognizer.Result())
                    if WAKE_WORD in result_json.get("text", ""):
                        print(f"\nWake word detected!")
                        if SYSTEM_PROMPT.strip():
                            conversation_history = [{'role': 'system', 'content': SYSTEM_PROMPT.strip()}]
                        else:
                            conversation_history = []
                        play_activation_sound()
                        current_state = STATE_LISTENING_FOR_COMMAND
                        voiced_frames = []
                        user_is_speaking = False
                        print("Listening for your command...")
                continue

            audio_chunk_np = np.frombuffer(audio_chunk_raw, dtype=np.int16).copy()
            
            # --- NEW: VAD Debugger Logic ---
            if VAD_DEBUG_MODE:
                # To get the confidence, we need to convert to a tensor and run the model directly
                audio_float32 = audio_chunk_np.astype(np.float32) / 32768.0
                confidence = vad_iterator.model(torch.from_numpy(audio_float32), INPUT_RATE).item()
                print(f"\rVAD Confidence: {confidence:.2f}", end='', flush=True)

            is_speech = confidence > VAD_THRESHOLD if VAD_DEBUG_MODE else (vad_iterator(audio_chunk_np) is not None)

            if is_speech:
                if not user_is_speaking:
                    user_is_speaking = True
                    if VAD_DEBUG_MODE: print() # Move to a new line after the confidence meter
                    print("I hear you speaking...", end='', flush=True)
                silence_start_time = None
                voiced_frames.append(audio_chunk_raw)
            elif user_is_speaking:
                if silence_start_time is None:
                    silence_start_time = time.time()
                
                if (time.time() - silence_start_time) > SILENCE_BEFORE_PROCESSING_SECONDS:
                    print("\nProcessing command...")
                    full_audio_np = np.frombuffer(b''.join(voiced_frames), dtype=np.int16).astype(np.float32) / 32768.0
                    segments, _ = stt_model.transcribe(full_audio_np, beam_size=5, language="en")
                    user_text = "".join(segment.text for segment in segments).strip()
                    
                    voiced_frames = []; silence_start_time = None; user_is_speaking = False
                    
                    if not user_text:
                        print("Heard nothing. Still listening...")
                        continue

                    print(f"User: {user_text}")

                    if END_CALL_PHRASE in user_text.lower():
                        print("End call phrase detected. Ending conversation.")
                        speak_text_system("Goodbye!", tts_engine)
                        conversation_history = []
                        current_state = STATE_WAITING_FOR_WAKE_WORD
                        print(f"\n--- Ready! Listening for '{WAKE_WORD}' ---")
                        continue
                    
                    assistant_response, conversation_history = ask_ollama(user_text, conversation_history, OLLAMA_OPTIONS)
                    speak_text_system(assistant_response, tts_engine)
                    
                    print("\nStill in conversation, listening for your reply...")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        p_mic.terminate()

if __name__ == "__main__":
    main()