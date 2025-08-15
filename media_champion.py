#!/usr/bin/env python
# media_champion.py - Unified offline and live media summariser using Whisper and OCI Generative AI.
#
# Prerequisites:
#   - Python 3.8 or higher
#   - ffmpeg in PATH (macOS: brew install ffmpeg; Ubuntu: sudo apt-get install ffmpeg)
#   - pip install openai-whisper oci
#   - For live recording: pip install sounddevice scipy numpy, plus PortAudio
#       macOS: brew install portaudio
#       Ubuntu: sudo apt-get install portaudio19-dev
#   - Ensure OCI CLI config is set up in ~/.oci/config
#
# Modes
#   --mode offline   Summarise media files. Input via --media-source or a list file
#   --mode live      Record from mic or use an existing transcript, then summarise
#
# Usage
#   Offline mode (single file)
#       python media_champion.py --mode offline --media-source ./talk.mp4
#
#   Offline mode (multiple files from a list file)
#       python media_champion.py --mode offline --media-source ./media.sources
#       python media_champion.py --mode offline               # uses ./media.sources by default
#
#   Live mode (record from microphone until Ctrl+C)
#       python media_champion.py --mode live
#
#   Live mode (summarise an existing transcript file)
#       python media_champion.py --mode live --use-transcript ./my_notes.txt
#
# Supported flags:
#   --mode {offline,live}    Operation mode
#   --output-dir DIR         Output directory
#   --output-base NAME       Optional base name for output files
#   --prompt PROMPT          Prompt name or path. 'default' uses built-in template
#   --prompt-name PROMPT     Same as --prompt
#   --media-source PATH      Media file or list file of media sources, one per line
#   --videos-file PATH       Same as --media-source
#   --max-tokens N           Maximum tokens for OCI summary (default: 1000)
#   --whisper-model MODEL    Whisper model size (default: base)
#   --oci-profile NAME       Override OCI profile from LoadConfig or env
#   --oci-model MODEL_ID     Override serving mode model id for OCI GenAI
#   --use-transcript FILE    Use an existing transcript file instead of recording (live mode)
#   --input-device INDEX     Input device index for live recording
#   --samplerate RATE        Sampling rate for live recording (default: 16000)
#   --channels N             Number of channels for live recording (default: 1)
#   --diarisation            Placeholder flag, accepted for compatibility
#   --timestamps             Placeholder flag, accepted for compatibility
#   --model                  Placeholder flag, accepted for compatibility
#   --chunking               Placeholder flag, accepted for compatibility
#   --log-level LEVEL        Logging level [debug,info,warning,error]
#   --model-args true|false  (removed) Previously allowed disabling model tuning args
#                           Model tuning arguments (max_tokens, temperature, frequency_penalty,
#                           presence_penalty, top_p, top_k) are included by default and cannot
#                           be disabled via CLI in this build.
#
# media.sources file format
#   One media path per line
#   Lines starting with '#' are ignored
#
# Prompt customisation (--prompt)
#   If omitted or set to "default", uses the built-in summarisation prompt
#   Otherwise searches for <prompt-name>.prompt in:
#       ./<prompt-name>.prompt
#       ./prompts/<prompt-name>.prompt
#       <script_dir>/<prompt-name>.prompt
#       <script_dir>/prompts/<prompt-name>.prompt
#       ~/prompts/<prompt-name>.prompt
#   You can also pass a full path to a .prompt file
#   If the prompt file contains {transcript}, it is replaced with the transcript text
#   Otherwise, the transcript is appended under a "Transcript:" section
#
# Output naming
#   Offline: <basename>.summary.md and <basename>.transcript.txt next to the input file
#            If --output-dir is passed, files go there instead
#   Live:    <base>.wav (recorded audio), <base>.summary.md and <base>.transcript.txt
#            Default folder is ./outputs unless --output-dir is passed
#   Batch offline runs only create a summary index when all outputs share one folder
#
# Examples
#   # Offline single video with default prompt
#   python media_champion.py --mode offline --media-source ./meeting.mp4
#
#   # Offline batch from media.sources with custom prompt
#   python media_champion.py --mode offline --prompt project_update
#
#   # Offline single file with specific Whisper model and max tokens
#   python media_champion.py --mode offline --media-source ./call.mp3 --whisper-model small --max-tokens 1500
#
#   # Offline, store summaries in a custom folder
#   python media_champion.py --mode offline --media-source ./media.sources --output-dir ./summaries
#
#   # Live record, save in custom folder with specific Whisper model
#   python media_champion.py --mode live --output-dir ./live_summaries --whisper-model medium
#
#   # Live record with a custom prompt stored in ./prompts
#   python media_champion.py --mode live --prompt slack_summary
#
#   # Summarise an existing transcript file with a prompt in ~/prompts
#   python media_champion.py --mode offline --use-transcript ./notes.txt --prompt meeting_analysis
#
#   # Offline single file, use a full path to prompt file
#   python media_champion.py --mode offline --media-source ./training.mp4 --prompt /path/to/custom.prompt
#
#   # Override OCI model serving mode
#   python media_champion.py --mode offline --media-source ./meeting.mp4 --oci-model ocid1.genaimodel.oc1..aaaa...

import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

import argparse
import itertools
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional

# Third party deps
import whisper
try:
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
    import numpy as np
except Exception:
    sd = None  # live mode will validate availability

# OCI GenAI
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

# Local config loader expected from existing projects
from load_config import LoadConfig

# =======================================================================================
# Centralised constants and defaults
# =======================================================================================

properties = LoadConfig()

BASE_DIR = Path(__file__).resolve().parent
CWD = Path.cwd()
HOME = Path.home()

# Names
MEDIA_SOURCES_NAME = "media.sources"
OUTPUT_DIR_NAME = "outputs"
PROMPT_DIR_NAME = "prompts"
SUMMARY_INDEX_NAME = "summary_index.md"
LIVE_BASENAME = "recording"
PROMPT_EXT = ".prompt"

# Resolved default paths
OCI_CONFIG_PATH = HOME / ".oci" / "config"
DEFAULT_SOURCES_FILE = CWD / MEDIA_SOURCES_NAME
DEFAULT_OUTPUT_DIR = CWD / OUTPUT_DIR_NAME
DEFAULT_PROMPT_DIR = CWD / PROMPT_DIR_NAME

# Where to look for prompts, in order
PROMPT_SEARCH_LOCATIONS = (
    CWD,
    DEFAULT_PROMPT_DIR,
    BASE_DIR,
    BASE_DIR / PROMPT_DIR_NAME,
    HOME / PROMPT_DIR_NAME,
)

AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".aac", ".ogg", ".opus"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

VALID_MEDIA_RE = re.compile(
    r".+\.(wav|m4a|mp3|flac|aac|ogg|opus|mp4|mov|mkv|avi|webm)$",
    re.IGNORECASE,
)

# =======================================================================================
# Logging
# =======================================================================================

def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

log = logging.getLogger("media_champion")

# =======================================================================================
# Small utils
# =======================================================================================

def pstr(p: Path) -> str:
    return str(p)

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# =======================================================================================
# OCI client
# =======================================================================================

def get_oci_client(oci_profile: Optional[str]) -> GenerativeAiInferenceClient:
    profile = oci_profile or os.getenv("OCI_PROFILE") or properties.getDefaultProfile()
    # Endpoint resolved from properties. No CLI flag for endpoint.
    endpoint = properties.getEndpoint()

    config = oci.config.from_file(pstr(OCI_CONFIG_PATH), profile)
    return GenerativeAiInferenceClient(
        config=config,
        service_endpoint=endpoint,
        retry_strategy=NoneRetryStrategy(),
        timeout=(10, 240),
    )

# =======================================================================================
# Prompt handling
# =======================================================================================

def _default_prompt(transcript: str) -> str:
    return (
        "You are a summarisation assistant. Carefully analyse the following transcript. First, determine the type of "
        "recording (e.g. meeting, presentation, interview, podcast, lecture, casual conversation). Then extract and "
        "summarise the key information with high coverage. Do not skip technical details, specific examples, or critical "
        "explanations.\n\n"
        "Before anything else, identify the participants in the call by analysing who is actively speaking. "
        "Ignore people who are only mentioned by others but do not speak directly.\n\n"
        "If the recording is a meeting, clearly identify:\n"
        "- All key discussion points (group them if needed)\n"
        "- Actions assigned (with owner, if mentioned). Highlight time sensitive or high priority items\n"
        "- Decisions made (with context)\n\n"
        "For all other types, include:\n"
        "- Participants\n"
        "- Type of Recording\n"
        "- Main topics covered (grouped logically)\n"
        "- Key insights and takeaways\n"
        "- Any actions or suggestions shared\n\n"
        "In all cases:\n"
        "- Summarise the overall sentiment or tone of the call. Note any shifts\n"
        "- Attribute points to speakers where identity is clear or implied\n"
        "- Note if any parts of the transcript are unclear, noisy, or incomplete\n"
        "- Use British spelling\n\n"
        "Structure the summary as follows: 1) Participants, 2) Type of Recording, 3) Sentiment, 4) Key Topics, 5) Actions & Owners, 6) Decisions, 7) Notable Quotes or Examples.\n\n"
        "Transcript:\n"
        f"{transcript.strip()}\n"
    )

def _find_prompt_file(prompt_value: str) -> Optional[Path]:
    v = prompt_value.strip()
    cand = Path(v)
    if cand.suffix.lower() != PROMPT_EXT:
        cand = cand.with_suffix(PROMPT_EXT)
    if Path(v).anchor or any(sep in v for sep in (os.sep, "/")):
        return cand if cand.is_file() else None
    names = [v, f"{v}{PROMPT_EXT}"]
    for base in PROMPT_SEARCH_LOCATIONS:
        for name in names:
            p = (base / name)
            if p.suffix.lower() != PROMPT_EXT:
                p = p.with_suffix(PROMPT_EXT)
            if p.is_file():
                return p
    return None

def build_prompt(transcript: str, prompt: str) -> str:
    if not prompt or prompt.strip().lower() == "default":
        return _default_prompt(transcript)
    path = _find_prompt_file(prompt)
    if not path:
        searched = ", ".join(pstr(p) for p in PROMPT_SEARCH_LOCATIONS)
        raise FileNotFoundError(
            f"Prompt file not found for '{prompt}'. Looked in: {searched}. "
            f"Pass a full path or a bare name without extension."
        )
    text = path.read_text(encoding="utf-8")
    return text.replace("{transcript}", transcript.strip()) if "{transcript}" in text \
           else f"{text.rstrip()}\n\nTranscript:\n{transcript.strip()}\n"

# =======================================================================================
# Transcription helpers
# =======================================================================================

def _is_compatible_wav(path: Path) -> bool:
    try:
        with wave.open(pstr(path), "rb") as wf:
            params = wf.getparams()
            return params.nchannels == 1 and params.sampwidth == 2 and params.framerate == 16000
    except Exception:
        return False

def _require_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        raise FileNotFoundError("ffmpeg not found in PATH. Install it first.")

def _ffmpeg_extract_audio(input_path: Path, out_wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", pstr(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        pstr(out_wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _prepare_audio_for_whisper(input_path: Path, tmpdir: Path) -> Tuple[Path, bool]:
    ext = input_path.suffix.lower()
    if ext == ".wav" and _is_compatible_wav(input_path):
        return input_path, False
    if ext in AUDIO_EXTS:
        return input_path, False
    out_wav = tmpdir / "audio.wav"
    _ffmpeg_extract_audio(input_path, out_wav)
    return out_wav, True

@lru_cache(maxsize=1)
def get_whisper_model(name: str = "base"):
    return whisper.load_model(name)

def transcribe_file(path: Path, whisper_model: str) -> Tuple[str, Optional[List[dict]]]:
    model = get_whisper_model(whisper_model)
    result = model.transcribe(pstr(path), fp16=False)
    text = result.get("text", "").strip()
    return text, result.get("segments")

def transcribe_media(input_path: Path, whisper_model: str) -> Tuple[str, Optional[List[dict]]]:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        audio_path, _ = _prepare_audio_for_whisper(input_path, tmpdir)
        return transcribe_file(audio_path, whisper_model)

# =======================================================================================
# Summarisation
# =======================================================================================

def summarise_text(
    client: GenerativeAiInferenceClient,
    transcript: str,
    prompt_value: str,
    *,
    max_tokens: int,
    oci_model: Optional[str] = None,
    compartment_id: Optional[str] = None,
    include_model_args: bool = True,
) -> str:
    prompt_text = build_prompt(transcript, prompt_value)
    content = TextContent(text=prompt_text)
    message = Message(role="USER", content=[content])
    # Include model tuning args by default. If include_model_args is False,
    # construct the request without explicit tuning parameters so the model's
    # defaults are used by the serving layer.
    if include_model_args:
        chat_request = GenericChatRequest(
            api_format=BaseChatRequest.API_FORMAT_GENERIC,
            messages=[message],
            max_tokens=max_tokens,
            temperature=0.25,
            frequency_penalty=1,
            presence_penalty=0,
            top_p=1,
            top_k=0,
        )
    else:
        chat_request = GenericChatRequest(
            api_format=BaseChatRequest.API_FORMAT_GENERIC,
            messages=[message],
        )
    serving_model_id = oci_model or properties.getModelOcid()
    chat_detail = ChatDetails(
        serving_mode=OnDemandServingMode(model_id=serving_model_id),
        chat_request=chat_request,
        compartment_id=compartment_id or properties.getCompartment(),
    )
    response = client.chat(chat_detail)
    return response.data.chat_response.choices[0].message.content[0].text.strip()

# =======================================================================================
# Output helpers
# =======================================================================================

def deterministic_paths(output_dir: Path, base: str) -> Tuple[Path, Path]:
    summary = output_dir / f"{base}.summary.md"
    transcript = output_dir / f"{base}.transcript.txt"
    return summary, transcript

def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")

def append_text(path: Path, content: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(content)

# =======================================================================================
# Source resolution for offline mode
# =======================================================================================

def _read_sources_file(list_path: Path) -> List[str]:
    items: List[str] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
    return items

def resolve_sources(media_source: Optional[str]) -> List[Path]:
    """
    Returns a list of absolute media paths.
    - If media_source is a media file, return it
    - If media_source is a text file, read paths from it
    - If media_source is None, try DEFAULT_SOURCES_FILE
    """
    if media_source is None:
        if not DEFAULT_SOURCES_FILE.exists():
            raise FileNotFoundError(
                f"No --media-source provided and {pstr(DEFAULT_SOURCES_FILE)} not found. "
                f"Create {MEDIA_SOURCES_NAME} with one path per line or pass --media-source PATH."
            )
        list_path = DEFAULT_SOURCES_FILE
        log.info("Using default sources list %s", pstr(list_path))
        sources = _read_sources_file(list_path)
    else:
        src = Path(os.path.expanduser(media_source))
        if not src.exists():
            raise FileNotFoundError(f"--media-source path not found: {pstr(src)}")
        if VALID_MEDIA_RE.match(pstr(src)):
            sources = [pstr(src)]
        else:
            sources = _read_sources_file(src)

    resolved: List[Path] = []
    for p in sources:
        ap = Path(os.path.expanduser(p)).resolve()
        if not ap.exists():
            log.warning("Skipping missing path: %s", p)
            continue
        if not VALID_MEDIA_RE.match(pstr(ap)):
            log.warning("Skipping non media path: %s", p)
            continue
        resolved.append(ap)

    if not resolved:
        raise ValueError("No valid media sources to process after filtering.")
    return resolved

# =======================================================================================
# Live recording helpers
# =======================================================================================

class LiveRecorder:
    """
    Minimal live recorder. Press Ctrl+C to stop.
    Frames are stored then written to wav.
    """
    def __init__(self, samplerate: int = 16000, channels: int = 1, device: Optional[int] = None):
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.frames: List["np.ndarray"] = []  # type: ignore

    def _callback(self, indata, frames_count, time_info, status):
        self.frames.append(indata.copy())

    def _spinner(self, stop_event: threading.Event):
        spinner = itertools.cycle(["⏱️", "🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚"])
        while not stop_event.is_set():
            sys.stdout.write(f"\rRecording... {next(spinner)}  Press Ctrl+C to stop.")
            sys.stdout.flush()
            time.sleep(0.3)

    def record_until_interrupt(self) -> "np.ndarray":  # type: ignore
        if sd is None:
            raise RuntimeError("sounddevice is not available. Install it to use live mode.")
        stop_event = threading.Event()
        spinner_t = threading.Thread(target=self._spinner, args=(stop_event,), daemon=True)
        spinner_t.start()
        try:
            with sd.InputStream(samplerate=self.samplerate, channels=self.channels, device=self.device, callback=self._callback):
                while True:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            spinner_t.join()
            print("\nStopped recording.")
        if not self.frames:
            raise RuntimeError("No audio captured. Check your input device settings.")
        data = np.concatenate(self.frames, axis=0)
        return data

    def save_wav(self, path: Path, data: "np.ndarray") -> None:  # type: ignore
        wav_write(pstr(path), self.samplerate, data)

# =======================================================================================
# CLI
# =======================================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Media Champion, unified offline batch and live summariser using Whisper and OCI Generative AI."
    )
    # Required mode
    p.add_argument("--mode", required=True, choices=["offline", "live"], help="Operation mode")

    # Shared
    p.add_argument("--output-dir", default=None,
                   help="Output directory. Offline default is the source file's folder. Live default is ./outputs")
    p.add_argument("--prompt", default="default",
                   help="Prompt name or path. 'default' uses built in prompt")
    p.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for the summary")
    p.add_argument("--whisper-model", default="base", help="Whisper model name, for example base, small, medium, large-v3")
    p.add_argument("--oci-profile", default=None, help="OCI profile to use, overrides LoadConfig. Env OCI_PROFILE also supported")
    p.add_argument("--oci-model", default=None, help="Override OCI Generative AI serving mode model id")

    # Back compat alias
    p.add_argument("--prompt-name", dest="prompt", help=argparse.SUPPRESS)

    # Media source (offline)
    p.add_argument("--media-source",
                   help=f"Media file or list file of media sources, one per line. If omitted in offline, use {pstr(DEFAULT_SOURCES_FILE)}")
    # Back compat alias
    p.add_argument("--videos-file", dest="media_source", help=argparse.SUPPRESS)

    # Output naming
    p.add_argument("--output-base", help="Optional base name override. For single source or live. Batch offline uses per source basename")

    # Live options
    p.add_argument("--use-transcript", help="Use an existing transcript file instead of recording (live mode)")
    p.add_argument("--input-device", type=int, help="Input device index for live recording")
    p.add_argument("--samplerate", type=int, default=16000, help="Sampling rate for live recording (default 16000)")
    p.add_argument("--channels", type=int, default=1, help="Number of channels for live recording (default 1)")

    # Legacy compatibility, accepted but ignored
    p.add_argument("--diarisation", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--timestamps", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", help=argparse.SUPPRESS)
    p.add_argument("--chunking", help=argparse.SUPPRESS)

    p.add_argument("--log-level", default="info", help="Logging level [debug,info,warning,error]")
    p.add_argument(
        "--no-model-args",
        action="store_true",
        help=(
            "If set, do not include model tuning arguments (max_tokens, temperature, "
            "frequency_penalty, presence_penalty, top_p, top_k) in the OCI request; "
            "defaults will be used by the model."
        ),
    )

    return p

def post_parse_warnings(args: argparse.Namespace) -> None:
    if getattr(args, "videos_file", None):
        log.warning("Flag --videos-file is deprecated, use --media-source instead.")
    if getattr(args, "prompt_name", None):
        log.warning("Flag --prompt-name is deprecated, use --prompt instead.")
    if args.mode == "offline" and args.output_base and not args.media_source:
        log.warning("Offline mode with multiple sources uses each source basename. --output-base is ignored for multiple files.")
    if getattr(args, "diarisation", False):
        log.warning("Diarisation flag accepted for compatibility but is not active in this build.")
    if getattr(args, "timestamps", False):
        log.warning("Timestamps flag accepted for compatibility but is not active in this build.")
    if getattr(args, "model", None):
        log.warning("Model selection flag accepted for compatibility but Whisper 'base' is used unless you pass --whisper-model.")
    if getattr(args, "chunking", None):
        log.warning("Chunking flag accepted for compatibility but is not active in this build.")
    if getattr(args, "no_model_args", False):
        log.info("Model tuning args will NOT be included in OCI requests (using model defaults).")
    else:
        log.debug("Model tuning args (max_tokens, temperature, frequency_penalty, presence_penalty, top_p, top_k) will be included in OCI requests by default.")

# =======================================================================================
# Mode runners
# =======================================================================================

def run_offline(args: argparse.Namespace) -> int:
    _require_ffmpeg()
    sources = resolve_sources(args.media_source)
    client = get_oci_client(args.oci_profile)

    processed_groups = {}  # outdir -> list of processed files

    multi = len(sources) > 1
    try:
        for idx, src in enumerate(sources, 1):
            # Decide output directory per file
            if args.output_dir:
                outdir = Path(args.output_dir).resolve()
            else:
                outdir = src.parent  # default: next to the source file
            ensure_output_dir(outdir)

            base = src.stem
            if not multi and args.output_base:
                base = args.output_base

            summary_path, transcript_path = deterministic_paths(outdir, base)

            log.info("[%d/%d] Transcribing: %s", idx, len(sources), pstr(src))
            transcript, _segments = transcribe_media(src, args.whisper_model)
            write_text(transcript_path, transcript)

            log.info("[%d/%d] Summarising: %s", idx, len(sources), pstr(src))
            summary = summarise_text(
                client,
                transcript,
                args.prompt,
                max_tokens=args.max_tokens,
                include_model_args=not getattr(args, "no_model_args", False),
                oci_model=args.oci_model,
                compartment_id=None,
            )

            footer = (
                "\n\n---\n"
                f"_Generated on {datetime.now().isoformat(timespec='seconds')} "
                f"with Whisper '{args.whisper_model}', max_tokens={args.max_tokens if not getattr(args, 'no_model_args', False) else 'default'}, prompt='{args.prompt}', "
                f"model_args_included={not getattr(args, 'no_model_args', False)}, "
                f"oci_model='{args.oci_model or properties.getModelOcid()}'._\n"
            )
            write_text(summary_path, summary + footer)

            log.info("Wrote transcript: %s", pstr(transcript_path))
            log.info("Wrote summary:   %s", pstr(summary_path))

            processed_groups.setdefault(outdir, []).append((src, summary_path, transcript_path))

    except KeyboardInterrupt:
        log.warning("Interrupted by user. Writing index files for completed items.")

    # Write index only for groups that have more than one processed file
    for outdir, items in processed_groups.items():
        if len(items) > 1:
            index_path = outdir / SUMMARY_INDEX_NAME
            write_text(index_path, "# Summary Index\n\n")
            for src, spath, tpath in items:
                rel_s = os.path.relpath(pstr(spath), pstr(outdir))
                rel_t = os.path.relpath(pstr(tpath), pstr(outdir))
                append_text(index_path, f"- Source: `{pstr(src)}`\n  - Summary: [{rel_s}]({rel_s})\n  - Transcript: [{rel_t}]({rel_t})\n")
            log.info("Wrote batch index: %s", pstr(index_path))

    return 0

def run_live(args: argparse.Namespace) -> int:
    # Decide output directory for live
    outdir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    ensure_output_dir(outdir)

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    base = args.output_base or f"{LIVE_BASENAME}-{ts}"
    summary_path, transcript_path = deterministic_paths(outdir, base)
    audio_path = outdir / f"{base}.wav"  # persist the audio for future offline runs

    # Option to use an existing transcript file
    if args.use_transcript:
        tfile = Path(os.path.expanduser(args.use_transcript)).resolve()
        if not tfile.is_file():
            raise FileNotFoundError(f"--use-transcript file not found: {pstr(tfile)}")
        log.info("Using existing transcript: %s", pstr(tfile))
        transcript = tfile.read_text(encoding="utf-8")
        write_text(transcript_path, transcript)
    else:
        if sd is None:
            raise RuntimeError("sounddevice is not installed. Install it to use live mode.")
        log.info("Starting live recording. Press Ctrl+C to stop.")
        rec = LiveRecorder(samplerate=args.samplerate, channels=args.channels, device=args.input_device)
        data = rec.record_until_interrupt()

        # Save audio first, then transcribe from the saved file
        rec.save_wav(audio_path, data)
        log.info("Saved live audio to: %s", pstr(audio_path))

        log.info("Transcribing live recording from saved audio...")
        transcript, _segments = transcribe_file(audio_path, args.whisper_model)
        write_text(transcript_path, transcript)
        log.info("Wrote transcript: %s", pstr(transcript_path))

    log.info("Generating summary with OCI Generative AI...")
    client = get_oci_client(args.oci_profile)
    summary = summarise_text(
        client,
        transcript,
        args.prompt,
        max_tokens=args.max_tokens,
        include_model_args=not getattr(args, "no_model_args", False),
        oci_model=args.oci_model,
        compartment_id=None,
    )
    footer = (
        "\n\n---\n"
        f"_Generated on {datetime.now().isoformat(timespec='seconds')} "
        f"with Whisper '{args.whisper_model}', max_tokens={args.max_tokens if not getattr(args, 'no_model_args', False) else 'default'}, prompt='{args.prompt}', "
        f"model_args_included={not getattr(args, 'no_model_args', False)}, "
        f"oci_model='{args.oci_model or properties.getModelOcid()}'._\n"
    )
    write_text(summary_path, summary + footer)
    log.info("Wrote summary:   %s", pstr(summary_path))
    return 0

# =======================================================================================
# Main
# =======================================================================================

def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    post_parse_warnings(args)

    try:
        if args.mode == "offline":
            return run_offline(args)
        elif args.mode == "live":
            return run_live(args)
        else:
            log.error("Unknown mode. Use --mode {offline,live}.")
            return 2
    except FileNotFoundError as e:
        log.error(str(e))
        return 1
    except subprocess.CalledProcessError:
        log.error("ffmpeg failed to process media. Check that ffmpeg is installed and the file is playable.")
        return 1
    except oci.exceptions.ServiceError as e:
        log.error("OCI service error: %s", e)
        return 1
    except Exception as e:
        log.error("Unexpected error: %s", e)
        return 1

if __name__ == "__main__":
    sys.exit(main())
