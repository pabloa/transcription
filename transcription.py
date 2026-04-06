#!/usr/bin/env python3
"""
Transcripción + Diarización de reuniones
=========================================
Combina Whisper (transcripción) con pyannote.audio (identificación de speakers).

Requisitos:
    pip install openai-whisper pyannote.audio

Antes de correrlo:
    1. Creá una cuenta en https://huggingface.co
    2. Aceptá los términos de uso de estos modelos (REQUERIDO):
       - https://huggingface.co/pyannote/speaker-diarization-3.1
       - https://huggingface.co/pyannote/speaker-diarization-community-1
       - https://huggingface.co/pyannote/segmentation-3.0
    3. Generá un token en https://huggingface.co/settings/tokens
    4. Pasalo como argumento --hf-token o seteá la variable HF_TOKEN

Uso:
    python transcription.py audio.mp3
    python transcription.py audio.mp3 --hf-token hf_XXXX
    python transcription.py audio.mp3 --model medium --language es
    python transcription.py audio.mp3 --num-speakers 4
    python transcription.py audio.mp3 --rename "SPEAKER_00=Pablo,SPEAKER_01=Taylor"
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path


def check_dependencies() -> None:
    """Verifica que ffmpeg esté instalado."""
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg no está instalado.", file=sys.stderr)
        print("\nInstálalo con:", file=sys.stderr)
        print("  macOS:  brew install ffmpeg", file=sys.stderr)
        print("  Ubuntu: sudo apt-get install ffmpeg", file=sys.stderr)
        print("  Fedora: sudo dnf install ffmpeg", file=sys.stderr)
        print("  Arch:   sudo pacman -S ffmpeg", file=sys.stderr)
        sys.exit(1)


def format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato HH:MM:SS."""
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if td.days > 0 or hours > 0:
        return f"{hours + td.days * 24:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def save_transcription_cache(cache_path: str, transcription: dict) -> None:
    """Guarda el resultado de la transcripción en caché."""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(transcription, f)


def load_transcription_cache(cache_path: str) -> dict:
    """Carga el resultado de la transcripción desde caché."""
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_diarization_cache(cache_path: str, diarization: object) -> None:
    """Guarda el resultado de la diarización en caché."""
    import pickle
    with open(cache_path, "wb") as f:
        pickle.dump(diarization, f)


def load_diarization_cache(cache_path: str) -> object:
    """Carga el resultado de la diarización desde caché."""
    import pickle
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def transcribe(audio_path: str, model_name: str, language: str) -> dict:
    """Transcribe audio con Whisper."""
    import torch
    import whisper

    print(f"\n{'='*60}")
    print(f"  PASO 1/3: Transcribiendo con Whisper ({model_name})")
    print(f"{'='*60}\n")

    # Force GPU usage
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Usando dispositivo: {device}")

    t0 = time.time()
    model = whisper.load_model(model_name, device=device)
    print(f"  Modelo cargado en {time.time() - t0:.1f}s")

    t0 = time.time()
    result = model.transcribe(audio_path, language=language, fp16=False)
    elapsed = time.time() - t0
    print(f"  Transcripción completada en {elapsed:.1f}s")
    print(f"  Segmentos detectados: {len(result['segments'])}")

    return result


def diarize(audio_path: str, hf_token: str, num_speakers: int = None) -> object:
    """Identifica speakers con pyannote."""
    import torch
    from pyannote.audio import Pipeline

    print(f"\n{'='*60}")
    print(f"  PASO 2/3: Identificando speakers con pyannote")
    print(f"{'='*60}\n")

    # Force GPU usage
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Usando dispositivo: {device}")

    t0 = time.time()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    pipeline.to(torch.device(device))
    print(f"  Pipeline cargada en {time.time() - t0:.1f}s")

    t0 = time.time()
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    output = pipeline(audio_path, **kwargs)
    diarization = output.speaker_diarization
    elapsed = time.time() - t0

    speakers = set()
    for _, _, label in diarization.itertracks(yield_label=True):
        speakers.add(label)

    print(f"  Diarización completada en {elapsed:.1f}s")
    print(f"  Speakers detectados: {len(speakers)} ({', '.join(sorted(speakers))})")

    return diarization


def assign_speakers(transcription: dict, diarization) -> list:
    """Asigna un speaker a cada segmento de transcripción."""
    segments = []

    for seg in transcription["segments"]:
        seg_start = seg["start"]
        seg_end = seg["end"]
        text = seg["text"].strip()

        if not text:
            continue

        # Buscar el speaker que más se superpone con este segmento
        best_speaker = "DESCONOCIDO"
        best_overlap = 0.0

        for turn, _, label in diarization.itertracks(yield_label=True):
            overlap_start = max(seg_start, turn.start)
            overlap_end = min(seg_end, turn.end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = label

        segments.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": best_speaker,
            "text": text,
        })

    return segments


def merge_consecutive(segments: list) -> list:
    """
    Mergea segmentos consecutivos del mismo speaker para
    producir bloques de texto más legibles.
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]
        # Si es el mismo speaker y hay menos de 2s de gap, mergear
        if seg["speaker"] == prev["speaker"] and (seg["start"] - prev["end"]) < 2.0:
            prev["end"] = seg["end"]
            prev["text"] += " " + seg["text"]
        else:
            merged.append(seg.copy())

    return merged


def rename_speakers(segments: list, rename_map: dict) -> list:
    """Renombra speakers según el mapa proporcionado."""
    for seg in segments:
        if seg["speaker"] in rename_map:
            seg["speaker"] = rename_map[seg["speaker"]]
    return segments


def export_txt(segments: list, output_path: str):
    """Exporta a texto plano legible."""
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            ts = format_timestamp(seg["start"])
            f.write(f"[{ts}] {seg['speaker']}:\n")
            f.write(f"  {seg['text']}\n\n")

    print(f"  Texto plano: {output_path}")


def export_srt(segments: list, output_path: str):
    """Exporta a formato SRT (subtítulos)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"[{seg['speaker']}] {seg['text']}\n\n")

    print(f"  Subtítulos SRT: {output_path}")


def format_srt_time(seconds: float) -> str:
    """Formato SRT: HH:MM:SS,mmm"""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_json(segments: list, output_path: str):
    """Exporta a JSON estructurado."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"  JSON: {output_path}")


def export_md(segments: list, output_path: str):
    """Exporta a Markdown."""
    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write("# Transcripción\n\n")

        # Speakers summary
        speakers = sorted(set(seg["speaker"] for seg in segments))
        f.write(f"**Speakers:** {', '.join(speakers)}\n\n")

        # Content
        f.write("---\n\n")
        for seg in segments:
            ts = format_timestamp(seg["start"])
            f.write(f"**[{ts}] {seg['speaker']}**\n\n")
            f.write(f"{seg['text']}\n\n")

    print(f"  Markdown: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribir y separar voces de una reunión"
    )
    parser.add_argument("audio", help="Archivo de audio (mp3, wav, m4a, etc.)")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Modelo de Whisper (default: small). 'medium' es más preciso pero más lento.",
    )
    parser.add_argument(
        "--language", default="es", help="Código de idioma (default: es)"
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Token de Hugging Face (o setear HF_TOKEN env var)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Número de speakers si lo sabés (mejora la precisión)",
    )
    parser.add_argument(
        "--rename",
        default=None,
        help='Renombrar speakers: "SPEAKER_00=Pablo,SPEAKER_01=Taylor"',
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida (default: mismo que el audio)",
    )

    args = parser.parse_args()

    # Verificar dependencias
    check_dependencies()

    # Validaciones
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: No se encuentra el archivo '{audio_path}'")
        sys.exit(1)

    if not args.hf_token:
        print("Error: Necesitás un token de Hugging Face.")
        print("  Opción 1: export HF_TOKEN=hf_XXXX")
        print("  Opción 2: --hf-token hf_XXXX")
        print()
        print("Para obtener uno:")
        print("  1. https://huggingface.co/settings/tokens")
        print("  2. Aceptá estos modelos (REQUERIDO):")
        print("     - https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("     - https://huggingface.co/pyannote/speaker-diarization-community-1")
        print("     - https://huggingface.co/pyannote/segmentation-3.0")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    transcription_cache_path = output_dir / f"{stem}_transcription_cache.json"
    diarization_cache_path = output_dir / f"{stem}_diarization_cache.pkl"

    # Parse rename map
    rename_map = {}
    if args.rename:
        for pair in args.rename.split(","):
            old, new = pair.strip().split("=")
            rename_map[old.strip()] = new.strip()

    # === Pipeline ===
    total_t0 = time.time()

    # 1. Transcribir (o cargar desde caché)
    if transcription_cache_path.exists():
        print(f"\n{'='*60}")
        print(f"  PASO 1/3: Cargando transcripción desde caché")
        print(f"{'='*60}\n")
        transcription = load_transcription_cache(str(transcription_cache_path))
        print(f"  Transcripción cargada desde caché")
        print(f"  Segmentos: {len(transcription['segments'])}")
        print(f"\n  💡 Para empezar desde el principio, eliminá los cachés:")
        print(f"     rm {transcription_cache_path} {diarization_cache_path}\n")
    else:
        transcription = transcribe(str(audio_path), args.model, args.language)
        save_transcription_cache(str(transcription_cache_path), transcription)

    # 2. Diarizar (o cargar desde caché)
    if diarization_cache_path.exists():
        print(f"\n{'='*60}")
        print(f"  PASO 2/3: Cargando diarización desde caché")
        print(f"{'='*60}\n")
        diarization = load_diarization_cache(str(diarization_cache_path))
        print(f"  Diarización cargada desde caché")
        speakers = set()
        for _, _, label in diarization.itertracks(yield_label=True):
            speakers.add(label)
        print(f"  Speakers detectados: {len(speakers)} ({', '.join(sorted(speakers))})")
        print(f"\n  💡 Para empezar desde el principio, eliminá los cachés:")
        print(f"     rm {transcription_cache_path} {diarization_cache_path}\n")
    else:
        diarization = diarize(str(audio_path), args.hf_token, args.num_speakers)
        save_diarization_cache(str(diarization_cache_path), diarization)

    # 3. Combinar
    print(f"\n{'='*60}")
    print(f"  PASO 3/3: Combinando y exportando")
    print(f"{'='*60}\n")

    segments = assign_speakers(transcription, diarization)
    segments = merge_consecutive(segments)

    if rename_map:
        segments = rename_speakers(segments, rename_map)
        print(f"  Speakers renombrados: {rename_map}")

    # Exportar en los 4 formatos
    export_txt(segments, str(output_dir / f"{stem}_transcripcion.txt"))
    export_srt(segments, str(output_dir / f"{stem}_transcripcion.srt"))
    export_json(segments, str(output_dir / f"{stem}_transcripcion.json"))
    export_md(segments, str(output_dir / f"{stem}_transcripcion.md"))

    total_elapsed = time.time() - total_t0

    # Resumen
    speakers = sorted(set(s["speaker"] for s in segments))
    print(f"\n{'='*60}")
    print(f"  LISTO")
    print(f"{'='*60}")
    print(f"  Tiempo total: {total_elapsed:.1f}s")
    print(f"  Speakers: {', '.join(speakers)}")
    print(f"  Bloques de texto: {len(segments)}")
    print()

    # Preview
    print("  Preview (primeros 10 bloques):")
    print(f"  {'-'*50}")
    for seg in segments[:10]:
        ts = format_timestamp(seg["start"])
        text = seg["text"][:80] + ("..." if len(seg["text"]) > 80 else "")
        print(f"  [{ts}] {seg['speaker']}: {text}")
    if len(segments) > 10:
        print(f"  ... y {len(segments) - 10} bloques más")
    print()


if __name__ == "__main__":
    main()
