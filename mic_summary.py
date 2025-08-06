# mic_summary.py - Record microphone audio, transcribe with Whisper, and summarize using OCI Generative AI.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - portaudio (macOS: brew install portaudio; Ubuntu: sudo apt-get install portaudio19-dev)
#   - pip3 install openai-whisper sounddevice scipy numpy oci
#   - Ensure OCI CLI config is set up in ~/.oci/config
#
# Usage:
#   python mic_summary.py --output-base <base name for output file name>
#   python mic_summary.py --use-transcript <path to an existing transcript file>
#   python mic_summary.py --output-dir <directory path to save output files>
#   python mic_summary.py --output-base <base name> --use-transcript <path/to/transcript.txt> --output-dir <output_directory>

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
import signal

stop_flag = threading.Event()

def handle_interrupt(signum, frame):
    stop_flag.set()

signal.signal(signal.SIGINT, handle_interrupt)

# dynamic load of configuration loader (load-config.py)
from load_config import LoadConfig

# Load properties
properties = LoadConfig()

# --- Recording settings ---
fs = 16000
channels = 1
frames = []

def get_oci_client():
    config = oci.config.from_file('~/.oci/config', properties.getDefaultProfile())
    return GenerativeAiInferenceClient(
        config=config,
        service_endpoint=properties.getEndpoint(),
        retry_strategy=NoneRetryStrategy(),
        timeout=(10, 240)
    )

def record_audio():
    print("🎤 Recording... Press Ctrl+C to stop.")

    def callback(indata, frames_count, time, status):
        frames.append(indata.copy())

    def show_spinner():
        spinner = itertools.cycle(["⏱️", "🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚"])
        while not stop_flag.is_set():
            sys.stdout.write(f"\rRecording... {next(spinner)} ")
            sys.stdout.flush()
            sd.sleep(300)

    spinner_thread = threading.Thread(target=show_spinner)
    spinner_thread.start()

    try:
        with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
            while not stop_flag.is_set():
                sd.sleep(200)  # Short sleep so we can check the stop_flag often
    finally:
        stop_flag.set()
        spinner_thread.join()
        print("\r🛑 Recording stopped.")

def save_audio(path):
    audio_data = np.concatenate(frames, axis=0)
    write(path, fs, audio_data)

def transcribe_audio(audio_path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    # Include speaker labels if possible
    segments = result.get("segments", [])
    transcript = ""
    for seg in segments:
        speaker = f"Speaker {seg.get('speaker', '?')}" if 'speaker' in seg else "[Unlabelled]"
        transcript += f"{speaker}: {seg['text'].strip()}\n"
    return transcript.strip() or result.get("text", "").strip()

def summarize_transcript(client, transcript: str) -> str:
    prompt_text = (
        "You are a summarisation assistant. Carefully analyse the following transcript. First, determine the type of recording "
        "(e.g. meeting, presentation, interview, podcast, lecture, casual conversation). Then extract and summarise the key information "
        "with high coverage. Do not skip technical details, specific examples, or critical explanations.\n\n"

        "Before anything else, identify the **participants in the call**. Do this by analysing who is actively speaking in the transcript. "
        "Ignore any individuals who are only mentioned or referenced by others but do not speak directly. "
        "List the names (or identifiers) of all speakers who contribute verbally to the conversation.\n\n"

        "If the recording is a meeting, clearly identify:\n"
        "- All key discussion points (group them if needed)\n"
        "- Actions assigned (with owner, if mentioned). Highlight time-sensitive or high-priority items\n"
        "- Decisions made (with context)\n\n"

        "For all other types, include:\n"
        "- Participants\n"
        "- Type of Recording\n"
        "- Main topics covered (grouped logically)\n"
        "- Key insights and takeaways\n"
        "- Any actions or suggestions shared\n\n"

        "In all cases:\n"
        "- Summarise the overall sentiment or tone of the call (e.g. collaborative, tense, enthusiastic, confused). Note any shifts in mood\n"
        "- Attribute points to speakers wherever identity is clear or implied. Avoid generic phrasing like 'someone said' if attribution can be inferred\n"
        "- Note if any parts of the transcript are unclear, noisy, or incomplete\n"
        "- Use British spelling\n\n"

        "Use clear section headers and bullet points. Ensure no critical point, insight, or decision is missed.\n\n"
        "Structure the summary as follows: 1) Participants, 2) Type of Recording, 3) Sentiment, 4) Key Topics, 5) Actions & Owners, 6) Decisions, 7) Notable Quotes or Examples.\n\n"

        "Transcript:\n"
        f"{transcript.strip()}\n\n"
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
    parser = argparse.ArgumentParser(description="Record audio or summarise existing transcript.")
    parser.add_argument("--output-base", type=str, default="recording", help="Base name for output files")
    parser.add_argument("--use-transcript", type=str, help="Path to an existing transcript file")
    parser.add_argument("--output-dir", type=str, default="./out", help="Directory to save output files")
    args = parser.parse_args()
    # If a transcript file is provided and output directory is default, use transcript file's directory
    if args.use_transcript and args.output_dir == './out':
        args.output_dir = os.path.dirname(os.path.abspath(args.use_transcript)) or '.'

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, f"{args.output_base}-{timestamp}")

    transcript_file = f"{base_path}-full_transcript.txt"
    summary_file = f"{base_path}-summary.txt"

    if args.use_transcript:
        transcript_path = os.path.abspath(args.use_transcript)
        # If default output directory is used, use the transcript file's directory
        if args.output_dir == "./out":
            output_dir = os.path.dirname(transcript_path) or '.'
        else:
            output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("📓 Using existing transcript...")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        # Derive summary file name from the input transcript's file name
        base_name, ext = os.path.splitext(os.path.basename(transcript_path))
        if "full_transcript" in base_name:
            summary_filename = base_name.replace("full_transcript", "summary") + ext
        else:
            summary_filename = base_name + "-summary" + ext
        summary_file = os.path.join(output_dir, summary_filename)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        base_path = os.path.join(output_dir, f"{args.output_base}-{timestamp}")
        transcript_file = f"{base_path}-full_transcript.txt"
        summary_file = f"{base_path}-summary.txt"

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "recording.wav")
            record_audio()
            save_audio(audio_path)
            print("🧠 Transcribing...")
            transcript = transcribe_audio(audio_path)
            write_file(transcript_file, transcript)

    print("📡 Generating summary with OCI Gen AI...")
    client = get_oci_client()
    summary = summarize_transcript(client, transcript)
    write_file(summary_file, summary)

    if not args.use_transcript:
        print(f"📝 Saved transcript to: {transcript_file}")
    print(f"✅ Saved summary to: {summary_file}")

if __name__ == "__main__":
    main()
