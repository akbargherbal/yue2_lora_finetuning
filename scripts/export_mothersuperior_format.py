#!/usr/bin/env python3
"""Export the built dataset to Mothersuperior's per-song format.

Mothersuperior's calibration scripts (``prep_real.py``, ``cursor_prep.py``,
``joint.py``) expect a flat corpus directory where each song is three files:

    <name>.flac        audio
    <name>.lyrics.txt  full lyrics, section tags intact
    <name>.txt         style caption, starting with a trigger phrase

Our dataset (``<split>/<name>.mp3`` + ``.lyrics.txt`` + ``.style.txt``) is
close but not identical, so this script transcribes it:

  - mp3 -> flac (lossless re-encode; the mp3 is already lossy, so this only
    satisfies their loader, it does not recover quality),
  - ``.lyrics.txt`` copied byte-for-byte,
  - ``<name>.txt`` = trigger phrase, a space, then the verbatim ``.style.txt``
    body.

It does **not** normalize lyrics section tags. It counts every ``[tag]`` it
finds and lists the ones that are not the simple lowercase form
(``[verse]``, ``[chorus]``, ...) so a mismatch against ``cursor_prep.py``'s
parser is visible before the full run, instead of silently rewritten.

Usage:
    python scripts/export_mothersuperior_format.py \\
        --dataset-root /content/data/dataset \\
        --out-root /content/ms_calib/corpus \\
        --trigger-phrase "maqamverse, in the style of maqamverse."
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

DEFAULT_SPLITS = ("train", "val")
AUDIO_SUFFIX = ".mp3"
TAG_RE = re.compile(r"\[([^\[\]]+)\]")
SIMPLE_TAG_RE = re.compile(r"^[a-z][a-z -]*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Built dataset root (contains train/ and val/).")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Flat output corpus directory.")
    p.add_argument("--trigger-phrase", required=True,
                   help="Style-caption trigger, e.g. 'maqamverse, in the style of maqamverse.'.")
    p.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                   help=f"Comma-separated split dirs to walk (default: {','.join(DEFAULT_SPLITS)}).")
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable (default: ffmpeg).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-encode every flac even if an up-to-date one exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written, write nothing.")
    return p.parse_args()


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("export_mothersuperior")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def compose_style(trigger: str, style_text: str) -> str:
    """Trigger phrase, one space, then the style body with outer whitespace trimmed."""
    return f"{trigger.strip()} {style_text.strip()}\n"


def audit_tags(lyrics: str) -> Counter:
    """Count every ``[tag]`` in the lyrics (case-sensitive, trimmed)."""
    return Counter(match.group(1).strip() for match in TAG_RE.finditer(lyrics))


def unsimple_tags(counts: Counter) -> list[str]:
    """Tags that are not the simple lowercase form Mothersuperior's examples use."""
    return sorted(tag for tag in counts if not SIMPLE_TAG_RE.fullmatch(tag))


def needs_encode(src: Path, dst: Path, overwrite: bool) -> bool:
    if overwrite or not dst.exists() or dst.stat().st_size == 0:
        return True
    return dst.stat().st_mtime < src.stat().st_mtime


def encode_flac(src: Path, dst: Path, ffmpeg: str, logger: logging.Logger) -> bool:
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vn", "-c:a", "flac", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg failed for %s: %s", src.name, (result.stderr or result.stdout).strip()[-500:])
        return False
    return True


def export(args: argparse.Namespace, logger: logging.Logger) -> int:
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)

    all_tags: Counter = Counter()
    seen: dict[str, str] = {}
    converted = skipped = failed = 0

    for split in splits:
        split_dir = args.dataset_root / split
        if not split_dir.is_dir():
            logger.warning("split dir missing, skipping: %s", split_dir)
            continue
        for src_audio in sorted(split_dir.glob(f"*{AUDIO_SUFFIX}")):
            base = src_audio.name[: -len(AUDIO_SUFFIX)]
            lyrics_src = split_dir / f"{base}.lyrics.txt"
            style_src = split_dir / f"{base}.style.txt"

            if base in seen:
                logger.error("duplicate track name %s (in %s and %s); output would collide",
                             base, seen[base], split)
                failed += 1
                continue
            seen[base] = split

            if not lyrics_src.is_file() or not style_src.is_file():
                logger.error("missing sidecar for %s: lyrics=%s style=%s",
                             base, lyrics_src.is_file(), style_src.is_file())
                failed += 1
                continue

            all_tags.update(audit_tags(lyrics_src.read_text(encoding="utf-8")))

            flac_dst = args.out_root / f"{base}.flac"
            lyrics_dst = args.out_root / f"{base}.lyrics.txt"
            style_dst = args.out_root / f"{base}.txt"

            if args.dry_run:
                logger.info("dry-run: %s -> %s (+ .lyrics.txt, .txt)", src_audio.name, flac_dst.name)
                continue

            if needs_encode(src_audio, flac_dst, args.overwrite):
                if not encode_flac(src_audio, flac_dst, args.ffmpeg, logger):
                    failed += 1
                    continue
                converted += 1
            else:
                skipped += 1

            shutil.copyfile(lyrics_src, lyrics_dst)
            style_dst.write_text(
                compose_style(args.trigger_phrase, style_src.read_text(encoding="utf-8")),
                encoding="utf-8",
            )

    logger.info("converted=%d skipped=%d failed=%d tracks=%d", converted, skipped, failed, len(seen))
    logger.info("lyrics section tags: %s", dict(all_tags.most_common()))
    suspect = unsimple_tags(all_tags)
    if suspect:
        logger.warning("tags not in simple lowercase form (verify against cursor_prep.py; NOT normalized): %s",
                       suspect)
    else:
        logger.info("all section tags are simple lowercase")

    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    logger = setup_logging()

    if not args.dataset_root.is_dir():
        logger.error("dataset root does not exist: %s", args.dataset_root)
        return 2
    if not args.dry_run and shutil.which(args.ffmpeg) is None:
        logger.error("ffmpeg not found: %s", args.ffmpeg)
        return 2

    logger.info("exporting %s -> %s (trigger=%r, dry-run=%s)",
                args.dataset_root, args.out_root, args.trigger_phrase, args.dry_run)
    return export(args, logger)


if __name__ == "__main__":
    sys.exit(main())
