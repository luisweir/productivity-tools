# video_summary_gen.py - Batch video summariser: reads videos listed in a file, transcribes audio using Whisper, and summarises the content using OCI Generative AI.
# Supports multiple summarisation styles via --prompt-name (e.g. default, slack_summary).
#
# Prerequisites:
#   - Python 3.7 or higher
#   - ffmpeg (brew install ffmpeg | apt-get install ffmpeg)
#   - pip3 install openai-whisper oci
#   - Ensure OCI CLI config is set up in ~/.oci/config
#
# Usage:
#   python video_summary_gen.py [--output-base BASE] [--output-dir DIR] [--videos-file VIDEOS_FILE] [--prompt-name PROMPT]
#
#   --output-base     Optional base name for output files (default: video filename)
#   --output-dir      Directory to save output files (default: current directory)
#   --videos-file     Path to file listing video paths (default: videos.txt)
#   --prompt-name     Prompt style to use: 'default' (structured summary) or 'slack_summary' (Slack-ready post). Default is 'default'.
#
#   Videos to process should be listed in the videos file, one video file path per line.

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
from load_config import LoadConfig
import argparse

# Load properties
properties = LoadConfig()

def get_oci_client():
    config = oci.config.from_file('~/.oci/config', properties.getDefaultProfile())
    return GenerativeAiInferenceClient(
        config=config,
        service_endpoint=properties.getEndpoint(),
        retry_strategy=NoneRetryStrategy(),
        timeout=(10, 240)
    )

def extract_audio(video_path: str, audio_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", audio_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def transcribe_audio(audio_path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result.get("text", "").strip()

def build_prompt(transcript: str, prompt_name: str) -> str:
    if prompt_name == "slack_summary":
        return (
            "# Goal\n"
            "Create a comprehensive Slack message based on the transcript that effectively summarizes a recorded call, "
            "highlighting key points and action items in a structured format that is easy for team members to digest.\n\n"
            "# Return Format\n"
            "A complete, ready-to-post Slack message that includes:\n"
            "- An engaging introduction with @channel mention\n"
            "- A brief description of the call's main focus\n"
            "- Recognition of key contributors\n"
            "- Clearly formatted key points and action items using emoji bullets\n"
            "- A professional closing statement\n"
            "- Use British spelling\n\n"
            "# Warnings\n"
            "- Ensure all participant names mentioned in the transcript are correctly tagged with @ symbols\n"
            "- Maintain a professional but friendly tone throughout the message\n"
            "- Don't include irrelevant details that might distract from the key takeaways\n"
            "- Avoid using technical jargon that might not be understood by all team members\n"
            "- Don't fabricate information not present in the transcript\n\n"
            "# Context\n"
            "- You are an team member responsible for sharing call recordings and summaries with your team\n"
            "- Your goal is to create a concise yet comprehensive Slack message that highlights the most important points from the call transcript, following the template format provided but customizing the content based on the actual transcript details\n"
            "- The message should be informative enough that team members who missed the call can quickly understand what was discussed and what actions they need to take\n"
            "- Your life depends on accurately extracting the most important information from the transcript and presenting it in a clear, structured format that follows the template but is customized to reflect the actual content of the call\n\n"
            "# Transcript\n"
            f"Transcript:\n{transcript.strip()}\n\n"
        )
    else:
        return (
            "You are a summarisation assistant. Carefully analyse the following transcript. First, determine the type of recording "
            "(e.g. meeting, presentation, interview, podcast, lecture, casual conversation). Then extract and summarise the key information "
            "with high coverage. Do not skip technical details, specific examples, or critical explanations.\n\n"
            "Before anything else, identify the **participants in the call**. Do this by analysing who is actively speaking in the transcript. "
            "Ignore any individuals who are only mentioned or referenced by others but do not speak directly.\n\n"
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

def summarize_transcript(client, transcript: str, prompt_name: str) -> str:
    prompt_text = build_prompt(transcript, prompt_name)
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

def run_summary(video_path: str, output_path: str, prompt_name: str):
    client = get_oci_client()
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, audio_path)
        transcript = transcribe_audio(audio_path)
        summary = summarize_transcript(client, transcript, prompt_name)
        write_output_file(output_path, summary, "Summary")
        base, ext = os.path.splitext(output_path)
        transcript_path = f"{base}-full_transcript{ext}"
        write_output_file(transcript_path, transcript, "Full transcript")
        return summary, transcript_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch video summarizer: read videos listed in a file, transcribe via Whisper, and summarize using OCI Generative AI."
    )
    parser.add_argument("--output-base", type=str, default=None, help="Base name for output files")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output files")
    parser.add_argument("--videos-file", type=str, default="videos.txt", help="Path to file listing video paths")
    parser.add_argument("--prompt-name", type=str, default="default", help="Name of the prompt to use (default or slack_summary)")
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
            summary, transcript_path = run_summary(video_path, output_file, args.prompt_name)
            print(f"Summary saved to {output_file}")
            print(f"Transcript saved to {transcript_path}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")