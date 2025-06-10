# mic-summary.py - Record microphone audio, transcribe with Whisper, and summarize using OCI Generative AI.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - portaudio (macOS: brew install portaudio; Ubuntu: sudo apt-get install portaudio19-dev)
#   - pip3 install openai-whisper sounddevice scipy numpy oci
#   - Ensure OCI CLI config is set up in ~/.oci/config
#
# Usage:
#   python mic-summary.py --output-base <base_name>

import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
import argparse
import os
import tempfile
import whisper
import threading
import itertools
import sys
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from datetime import datetime
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails, TextContent, Message, GenericChatRequest,
    OnDemandServingMode, BaseChatRequest
)
from oci.retry import NoneRetryStrategy
from LoadProperties import LoadProperties

# Load properties
properties = LoadProperties()

# --- Recording settings ---
fs = 16000
channels = 1
frames = []

def get_oci_client():
    config = oci.config.from_file('~/.oci/config', properties.getDefaultProfile())
    return GenerativeAiInferenceClient(
        config=config,
        service_endpoint=properties.getEndpoint(),  # If endpoint is a property, no parentheses
        retry_strategy=NoneRetryStrategy(),
        timeout=(10, 240)
    )

def record_audio():
    print("🎙️ Recording... Press Ctrl+C to stop.")

    stop_flag = threading.Event()

    def callback(indata, frames_count, time, status):
        frames.append(indata.copy())

    def show_spinner():
        spinner = itertools.cycle(["⏱️", "🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚"])
        while not stop_flag.is_set():
            sys.stdout.write(f"\rRecording... {next(spinner)} ")
            sys.stdout.flush()
            sd.sleep(300)

    try:
        spinner_thread = threading.Thread(target=show_spinner)
        spinner_thread.start()

        with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        stop_flag.set()
        spinner_thread.join()
        print("\r🛑 Recording stopped.")

def save_audio(path):
    audio_data = np.concatenate(frames, axis=0)
    write(path, fs, audio_data)

def transcribe_audio(audio_path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result.get("text", "").strip()

def summarize_transcript(client, transcript: str) -> str:
    prompt_text = (
        "You are a summarization assistant. Analyze the following transcript to determine what kind of recording it is "
        "(e.g., a meeting, lecture, casual conversation, podcast). If it's a meeting, include action items and decisions made. "
        "Otherwise, summarize appropriately.\n\n"
        "Transcript:\n"
        f"{transcript}\n\n"
        "Provide a summary with clear sections:\n"
        "- Type of Recording\n"
        "- Key Discussion Points\n"
        "- Action Items (if applicable)\n"
        "- Decisions Made (if applicable)\n\n"
        "Use bullet points. Keep it clear and structured."
    )

    content = TextContent(text=prompt_text)
    message = Message(role="USER", content=[content])
    chat_request = GenericChatRequest(
        api_format=BaseChatRequest.API_FORMAT_GENERIC,
        messages=[message],
        max_tokens=1000,
        temperature=0.3,
        frequency_penalty=1,
        presence_penalty=0,
        top_p=1,
        top_k=0
    )
    chat_detail = ChatDetails(
        serving_mode=OnDemandServingMode(model_id=properties.getModelOcid()),
        chat_request=chat_request,
        compartment_id=properties.getCompartment()
    )
    response = client.chat(chat_detail)
    return response.data.chat_response.choices[0].message.content[0].text.strip()

def write_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="Record microphone audio and generate transcript + summary.")
    parser.add_argument("--output-base", type=str, default="recording", help="Base name for output files (no extension)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    base_path = f"{args.output_base}-{timestamp}"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "recording.wav")

        record_audio()
        save_audio(audio_path)

        print("🧠 Transcribing...")
        transcript = transcribe_audio(audio_path)

        print("📡 Generating summary with OCI Gen AI...")
        client = get_oci_client()
        summary = summarize_transcript(client, transcript)

        transcript_file = f"{base_path}-full_transcript.txt"
        summary_file = f"{base_path}-summary.txt"

        write_file(transcript_file, transcript)
        write_file(summary_file, summary)

        print(f"\n✅ Saved summary to: {summary_file}")
        print(f"📝 Saved transcript to: {transcript_file}")

if __name__ == "__main__":
    main()
