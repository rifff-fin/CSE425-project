from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download timestamped MusicCaps clips from their YouTube IDs.")
    parser.add_argument("--captions", type=Path, default=Path("data/raw/musiccaps/musiccaps-public.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/musiccaps/audio"))
    parser.add_argument("--limit", type=int, default=100, help="Number of clips; use 0 for all available records.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    captions = args.captions if args.captions.is_absolute() else root / args.captions
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    yt_dlp = shutil.which("yt-dlp")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = None
    if not yt_dlp:
        raise RuntimeError("yt-dlp is required. Install it with: python -m pip install yt-dlp")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required. Install imageio-ffmpeg or add ffmpeg to PATH.")

    records = list(load_records(captions))[args.start_index:]
    if args.limit > 0:
        records = records[: args.limit]
    output_root.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    for index, record in enumerate(records, start=args.start_index):
        ytid = (record.get("ytid") or "").strip()
        if not ytid:
            skipped += 1
            continue
        target = output_root / f"{ytid}.wav"
        if target.exists() and not args.overwrite:
            downloaded += 1
            continue
        command = [
            yt_dlp, "--no-playlist", "--quiet", "--no-warnings", "--ffmpeg-location", ffmpeg, "--extract-audio",
            "--audio-format", "wav", "--postprocessor-args", "-ar 22050 -ac 1",
            "--download-sections", f"*{record['start_s']}-{record['end_s']}",
            "-o", str(output_root / f"{ytid}.%(ext)s"),
            f"https://www.youtube.com/watch?v={ytid}",
        ]
        result = subprocess.run(command, check=False)
        if result.returncode == 0 and target.exists():
            downloaded += 1
            print(f"[{index}] downloaded {ytid}")
        else:
            skipped += 1
            print(f"[{index}] unavailable {ytid}")
    print(f"Downloaded or already present: {downloaded}; unavailable: {skipped}")


if __name__ == "__main__":
    main()