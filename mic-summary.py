# mic-summary.py - Enhanced: Accepts existing transcript and attempts speaker diarisation.

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
        service_endpoint=properties.getEndpoint(),
        retry_strategy=NoneRetryStrategy(),
        timeout=(10, 240)
    )

def record_audio():
    print("🎤 Recording... Press Ctrl+C to stop.")

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

        "If the recording is a meeting, clearly identify:\n"
        "- All key discussion points (group them if needed)\n"
        "- Actions assigned (with owner, if mentioned). Highlight time-sensitive or high-priority items\n"
        "- Decisions made (with context)\n\n"

        "For all other types, include:\n"
        "- Type of Recording\n"
        "- Main topics covered (grouped logically)\n"
        "- Key insights and takeaways\n"
        "- Any actions or suggestions shared\n\n"

        "In all cases:\n"
        "- Summarise the overall sentiment or tone of the call (e.g. collaborative, tense, enthusiastic, confused). Note any shifts in mood\n"
        "- Attribute points to speakers wherever identity is clear or implied. Avoid generic phrasing like 'someone said' if attribution can be inferred\n"
        "- Note if any parts of the transcript are unclear, noisy, or incomplete\n\n"

        "Use clear section headers and bullet points. Ensure no critical point, insight, or decision is missed.\n\n"
        "Structure the summary as follows: 1) Type of Recording, 2) Sentiment, 3) Key Topics, 4) Actions & Owners, 5) Decisions, 6) Notable Quotes or Examples.\n\n"

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

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, f"{args.output_base}-{timestamp}")

    transcript_file = f"{base_path}-full_transcript.txt"
    summary_file = f"{base_path}-summary.txt"

    if args.use_transcript:
        print("📓 Using existing transcript...")
        with open(args.use_transcript, "r", encoding="utf-8") as f:
            transcript = f.read()
    else:
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
