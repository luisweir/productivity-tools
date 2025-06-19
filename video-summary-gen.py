# video-summary-gen.py - Batch video summarizer: read videos listed in videos.txt, transcribe via Whisper, and summarize using OCI Generative AI.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - ffmpeg (brew install ffmpeg | apt-get install ffmpeg)
#   - pip3 install openai-whisper oci
#   - Ensure OCI CLI config is set up in ~/.oci/config
#
# Usage:
#   python video-summary-gen.py
#   # Videos to process should be listed in videos.txt, one video file path per line.

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
from LoadProperties import LoadProperties

# Load properties
properties = LoadProperties()

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
        "Here is the transcript of a meeting call:\n\n"
        f"{transcript}\n\n"
        "Summarise the meeting so I can get a clear picture without listening to the recording.\n\n"
        "Break it down into:\n"
        "1) Key discussion points: Focus on the main topics covered. Highlight any issues raised or heated debate.\n"
        "2) Action items: List tasks or follow-ups, who is responsible (if known), and any open points.\n"
        "3) Decisions made: Clearly state what was agreed upon.\n\n"
        "Use bullet points for each section. Keep it concise and formal. If speaker attribution is possible, include it."
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

# Batch processing for videos listed in videos.txt
if __name__ == "__main__":
    try:
        with open("videos.txt", "r") as f:
            videos = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("videos.txt not found. Please create a file 'videos.txt' with one video path per line.")
        exit(1)

    for video_path in videos:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_file = f"{base}-summary.txt"
        print(f"Processing {video_path} -> {output_file}")
        try:
            summary, transcript_path = run_summary(video_path, output_file)
            print(f"Summary saved to {output_file}")
            print(f"Transcript saved to {transcript_path}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")



