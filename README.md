# Transcription and Diarization

Diarization and transcription of meetings using OpenAI Whisper and pyannote.audio.

## Requirements

- Python >= 3.13
- `openai-whisper`
- `pyannote-audio`
- A Hugging Face account and token.

## Setup

1. Create an account at [Hugging Face](https://huggingface.co).
2. Accept the terms of use for the following models:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Generate a token at [Hugging Face Settings](https://huggingface.co/settings/tokens).
4. Provide the token via the `--hf-token` argument or by setting the `HF_TOKEN` environment variable.

## Installation

You can install the dependencies using `pip`:

```bash
pip install openai-whisper pyannote.audio
```

Or using `uv`:

```bash
uv sync
```

## Usage

Basic usage:
```bash
python transcription.py audio.mp3
```

With Hugging Face token:
```bash
python transcription.py audio.mp3 --hf-token hf_XXXX
```

Specify model and language:
```bash
python transcription.py audio.mp3 --model medium --language es
```

Specify the number of speakers:
```bash
python transcription.py audio.mp3 --num-speakers 4
```

Rename speakers:
```bash
python transcription.py audio.mp3 --rename "SPEAKER_00=Pablo,SPEAKER_01=Taylor"
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
