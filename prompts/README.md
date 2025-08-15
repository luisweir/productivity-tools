Using custom prompts with --prompt-name

The media_champion.py script supports the --prompt-name option which allows you to supply custom prompt text to control how transcripts are summarised.

How it works
- If --prompt-name is omitted or set to "default", a built-in prompt is used.
- If you supply any other value, the script will try to load a file named <prompt-name>.prompt from a set of candidate locations:
  - ./<prompt-name>.prompt
  - ./prompts/<prompt-name>.prompt
  - <script_dir>/<prompt-name>.prompt
  - <script_dir>/prompts/<prompt-name>.prompt
  - ~/prompts/<prompt-name>.prompt
  - Or an explicit path (absolute or relative) you provide; the .prompt extension is optional.

Prompt file behaviour
- If the prompt file contains the placeholder {transcript}, it will be replaced with the actual transcript text.
- If the file does not contain {transcript}, the transcript will be appended under a "Transcript:" section.

Example
1. An example prompt is provided at prompts/slack_summary.prompt. Use the prompt name "slack_summary" to reference it.

2. Run media_champion.py for audio or video files (or a list of files):
   python media_champion.py --media-list videos.txt --prompt-name ./prompts/slack_summary

3. Run media_champion.py using an existing transcript and the example prompt:
   python media_champion.py --use-transcript /path/to/transcript.txt --prompt-name ./prompts/slack_summary

This makes it easy to experiment with different summarisation styles or constraints without changing the scripts.
