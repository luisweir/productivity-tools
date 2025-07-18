# video_summary_gen.py - Batch video summarizer: read videos listed in videos.txt, transcribe via Whisper, and summarize using OCI Generative AI.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - ffmpeg (brew install ffmpeg | apt-get install ffmpeg)
#   - pip3 install openai-whisper oci
#   - Ensure OCI CLI config is set up in ~/.oci/config
#
# Usage:
#   python video_summary_gen.py [--output-base BASE] [--output-dir DIR] [--videos-file VIDEOS_FILE]
#   # Videos to process should be listed in the videos file (default videos.txt), one video file path per line.

import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
import os
import subprocess
import tempfile
import whisper
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails,
    TextContent,
    Message,
    GenericChatRequest,
    OnDemandServingMode,
    BaseChatRequest,
)
from oci.retry import NoneRetryStrategy
# dynamic load of configuration loader (load-config.py)
from load_config import LoadConfig
import argparse

# Load properties
properties = LoadConfig()

def get_oci_client():
    config = oci.config.from_file('~/.oci/config', properties.getDefaultProfile())
    return GenerativeAiInferenceClient(
        config=config,
        service_endpoint=properties.getEndpoint(),  # If endpoint is a property, no parentheses
        retry_strategy=NoneRetryStrategy(),
        timeout=(10, 240)
    )


def extract_audio(video_path: str, audio_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ac", "1",
            "-ar", "16000",
            audio_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def transcribe_audio(audio_path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result.get("text", "").strip()


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
        "- Note if any parts of the transcript are unclear, noisy, or incomplete\n\n"

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
        max_tokens=800,
        temperature=0.25,
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


def write_output_file(path: str, content: str, label: str):
    with open(path, "w") as f:
        f.write(content)


def run_summary(video_path: str, output_path: str):
    client = get_oci_client()
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, audio_path)
        transcript = transcribe_audio(audio_path)
        summary = summarize_transcript(client, transcript)
        write_output_file(output_path, summary, "Summary")
        base, ext = os.path.splitext(output_path)
        transcript_path = f"{base}-full_transcript{ext}"
        write_output_file(transcript_path, transcript, "Full transcript")
        return summary, transcript_path

# Batch processing for videos listed in a file (default videos.txt)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch video summarizer: read videos listed in a file, transcribe via Whisper, and summarize using OCI Generative AI."
    )
    parser.add_argument(
        "--output-base", type=str, default=None,
        help="Base name for output files (default: use video filename)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--videos-file", type=str, default="videos.txt",
        help="Path to file listing video paths"
    )
    args = parser.parse_args()

    try:
        with open(args.videos_file, "r") as f:
            videos = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"{args.videos_file} not found. Please create it with one video path per line.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    for video_path in videos:
        default_base = os.path.splitext(os.path.basename(video_path))[0]
        base = args.output_base if args.output_base else default_base
        output_file = os.path.join(args.output_dir, f"{base}-summary.txt")
        print(f"Processing {video_path} -> {output_file}")
        try:
            summary, transcript_path = run_summary(video_path, output_file)
            print(f"Summary saved to {output_file}")
            print(f"Transcript saved to {transcript_path}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
