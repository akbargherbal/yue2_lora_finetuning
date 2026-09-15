#!/usr/bin/env python3
"""
prepare_dataset.py (v2 — matches the real min_4stars_ai_music/ layout)
------------------------------------------------------------------------
Layout assumed:

    <dataset_root>/<maqam>/<workspace_name>/
        workspace_manifest.json   # ALL tracks ever generated in this workspace
        some_title_SONG_A.mp3     # only the 4-5* survivors are physically present
        ...

Key ideas, matching how this corpus was actually built:

  - The manifest is a superset. A track entry with no matching audio file in
    its own workspace folder was rated below 4 stars and discarded — that's
    the filter, not a separate "status" field. We treat file-existence as
    ground truth for "keep".
  - Multiple takes of the same poem that ALL survived curation (SONG_A,
    SONG_B, SONG_C, RETAKE_SONG_A, ...) are NOT deduped down to one. If they
    all passed your 4-5* bar, they're legitimate audio diversity for the same
    caption/style target, not noise — keep them all by default (see
    --max-per-song if you want a cap).
  - The same poem can recur across *different* workspace folders (e.g. a poem
    regenerated on a later date, sometimes with a slightly different-cased
    workspace name). These are grouped into one "song group" by normalized
    title + maqam, regardless of which workspace or filename suffix they came
    from, so train/val splitting never leaks the same lyrics across the split.
  - maqam is taken from the top-level folder name (ajam/hijaz/kurd/nahawand),
    not re-derived from the caption text — the folder is authoritative here.
    The caption's own "Maqam X" mention is cross-checked as an integrity
    check and any mismatch is reported, not silently trusted either way.

Usage:
    python prepare_dataset.py --dataset-root ./min_4stars_ai_music \\
        --out-dir ./dataset --dry-run

    python prepare_dataset.py --dataset-root ./min_4stars_ai_music \\
        --out-dir ./dataset --max-per-song 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from maqam_prompt_generator import MAQAMS, build_prompt
except ImportError:
    print(
        "ERROR: could not import maqam_prompt_generator.py. "
        "Place it next to this script or fix the sys.path insert above.",
        file=sys.stderr,
    )
    raise

CANONICAL_MAQAMS = {name.lower(): name for name in MAQAMS.values()}


def nfc(text: str) -> str:
    """Normalize Unicode form so Arabic text compares equal regardless of
    whether it was composed/decomposed differently by whatever tool wrote
    the manifest vs. whatever tool wrote the filename to disk."""
    return unicodedata.normalize("NFC", text)


def normalize_title_key(title: str) -> str:
    """Collapse whitespace and normalize Unicode so the same poem title
    written slightly differently (extra space, different dash) still groups
    together across workspaces."""
    t = nfc(title).strip()
    t = re.sub(r"\s+", " ", t)
    return t


@dataclass
class Track:
    clip_id: str
    workspace: str
    maqam: str  # from folder, authoritative
    caption_maqam: str | None  # parsed from styles text, for cross-check
    original_title: str
    audio_path: Path
    styles: str
    start_phrase: str | None
    mood: str | None
    group_key: str = field(default="")
    caption: str | None = None


def extract_caption_maqam(styles: str) -> str | None:
    m = re.search(r"Maqam\s+([A-Za-z]+)", styles)
    if not m:
        return None
    return CANONICAL_MAQAMS.get(m.group(1).strip().lower())


def extract_start_phrase(styles: str) -> str | None:
    m = re.search(r'\[START_ON:\s*"([^"]+)"\]', styles)
    return m.group(1) if m else None


def extract_mood(styles: str) -> str | None:
    m = re.search(r'mood:\s*"([^"]+)"', styles)
    return m.group(1) if m else None


def find_local_audio(workspace_dir: Path, assigned_filename: str) -> Path | None:
    """Look for assigned_filename inside its own workspace folder, tolerant
    of Unicode-normalization mismatches between the manifest string and the
    filename actually on disk."""
    direct = workspace_dir / assigned_filename
    if direct.exists():
        return direct

    target_norm = nfc(assigned_filename)
    for candidate in workspace_dir.iterdir():
        if candidate.is_file() and nfc(candidate.name) == target_norm:
            return candidate
    return None


def walk_corpus(dataset_root: Path) -> list[Track]:
    tracks: list[Track] = []
    maqam_mismatches = []
    missing_audio_count = 0
    unknown_maqam_dirs = []

    for maqam_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        folder_maqam = CANONICAL_MAQAMS.get(maqam_dir.name.strip().lower())
        if folder_maqam is None:
            unknown_maqam_dirs.append(maqam_dir.name)
            continue

        for manifest_path in sorted(maqam_dir.rglob("workspace_manifest.json")):
            workspace_dir = manifest_path.parent
            data = json.loads(manifest_path.read_text(encoding="utf-8"))

            for rt in data.get("tracks", []):
                assigned_filename = rt["assigned_filename"]
                audio_path = find_local_audio(workspace_dir, assigned_filename)
                if audio_path is None:
                    missing_audio_count += 1  # filtered out below 4 stars
                    continue

                styles = rt.get("styles", "")
                caption_maqam = extract_caption_maqam(styles)
                if caption_maqam and caption_maqam != folder_maqam:
                    maqam_mismatches.append(
                        (str(audio_path), folder_maqam, caption_maqam)
                    )

                track = Track(
                    clip_id=rt["clip_id"],
                    workspace=workspace_dir.name,
                    maqam=folder_maqam,
                    caption_maqam=caption_maqam,
                    original_title=rt["original_title"],
                    audio_path=audio_path,
                    styles=styles,
                    start_phrase=extract_start_phrase(styles),
                    mood=extract_mood(styles),
                )
                track.group_key = f"{folder_maqam}::{normalize_title_key(track.original_title)}"
                tracks.append(track)

    if unknown_maqam_dirs:
        print(f"[warn] top-level folders not recognized as a maqam name, skipped: {unknown_maqam_dirs}")
    if maqam_mismatches:
        print(f"[warn] {len(maqam_mismatches)} tracks where the folder's maqam and the "
              f"caption's stated maqam disagree — folder wins, but check these:")
        for path, folder_m, caption_m in maqam_mismatches[:10]:
            print(f"    {path}: folder={folder_m} caption={caption_m}")
    print(f"Filtered out (in manifest, no matching audio file — below your rating bar): {missing_audio_count}")

    return tracks


def cap_per_song(tracks: list[Track], max_per_song: int | None) -> list[Track]:
    if max_per_song is None:
        return tracks
    groups: dict[str, list[Track]] = defaultdict(list)
    for t in tracks:
        groups[t.group_key].append(t)
    kept = []
    for group in groups.values():
        group.sort(key=lambda t: t.clip_id)  # deterministic, not quality-based
        kept.extend(group[:max_per_song])
    return kept


def render_captions(tracks: list[Track], include_mood: bool, keep_header: bool) -> None:
    for t in tracks:
        prompt_block = build_prompt(
            maqam_name=t.maqam,
            start_phrase=t.start_phrase,
            mood=t.mood if include_mood else None,
            instrumental=False,
        )
        if keep_header:
            t.caption = prompt_block
        else:
            # Keep only the actual field lines (genre/vocals/production/
            # instrumentation/mood), dropping the PROMPT:/EXCLUDE: wrapper,
            # code fences, and control-token header entirely.
            field_prefixes = ("genre:", "vocals:", "production:", "instrumentation:", "mood:")
            body_lines = [
                l for l in prompt_block.splitlines()
                if l.startswith(field_prefixes)
            ]
            t.caption = "\n".join(body_lines).strip()


def split_train_val(tracks: list[Track], val_fraction: float, seed: int) -> tuple[list[Track], list[Track]]:
    import random

    rng = random.Random(seed)
    groups_by_maqam: dict[str, list[str]] = defaultdict(list)
    seen_groups: dict[str, str] = {}
    for t in tracks:
        seen_groups[t.group_key] = t.maqam
    for group_key, maqam in seen_groups.items():
        groups_by_maqam[maqam].append(group_key)

    val_groups: set[str] = set()
    for maqam, group_keys in groups_by_maqam.items():
        rng.shuffle(group_keys)
        n_val = max(1, round(len(group_keys) * val_fraction)) if group_keys else 0
        val_groups.update(group_keys[:n_val])

    train, val = [], []
    for t in tracks:
        (val if t.group_key in val_groups else train).append(t)
    return train, val


def ascii_safe_slug(text: str, max_len: int = 40) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return slug[:max_len] or "track"


def write_split(tracks: list[Track], out_dir: Path, dry_run: bool) -> list[dict]:
    rows = []
    # BUGFIX: ascii_safe_slug(original_title) strips all Arabic characters,
    # so unrelated poems whose titles happen to share a leading number (or
    # have no digit at all) were collapsing onto the same slug -- e.g. both
    # "01-الوداع-..." and "01-عزة-الفارس-..." became just "01". That made
    # dest_audio collide across different poems, and shutil.copy2() below
    # silently overwrote earlier files with later ones. Assigning each
    # distinct group_key its own sequential, ASCII-safe poem_id guarantees
    # a unique destination filename regardless of what survives ASCII
    # stripping.
    poem_ids: dict[str, int] = {}
    next_poem_id = 1
    group_counters: dict[str, int] = defaultdict(int)
    for t in sorted(tracks, key=lambda t: (t.maqam, t.group_key, t.clip_id)):
        if t.group_key not in poem_ids:
            poem_ids[t.group_key] = next_poem_id
            next_poem_id += 1
        group_counters[t.group_key] += 1
        take_no = group_counters[t.group_key]
        slug = ascii_safe_slug(t.original_title)
        base = f"{t.maqam.lower()}_{poem_ids[t.group_key]:04d}_{slug}_take{take_no:02d}"
        dest_audio = out_dir / f"{base}{t.audio_path.suffix}"
        dest_caption = out_dir / f"{base}.txt"

        rows.append({
            "clip_id": t.clip_id,
            "workspace": t.workspace,
            "original_title": t.original_title,
            "maqam": t.maqam,
            "mood": t.mood or "",
            "take_no_in_group": take_no,
            "source_audio": str(t.audio_path),
            "dest_audio": str(dest_audio),
            "dest_caption": str(dest_caption),
        })

        if dry_run:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(t.audio_path, dest_audio)
        dest_caption.write_text(t.caption, encoding="utf-8")

    return rows


def print_top_repeated_songs(tracks: list[Track], top_n: int = 10) -> None:
    counts: dict[str, int] = defaultdict(int)
    titles: dict[str, str] = {}
    for t in tracks:
        counts[t.group_key] += 1
        titles[t.group_key] = f"{t.maqam}: {t.original_title}"
    most_common = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    print(f"\nMost-retried poems (surviving takes per song, top {top_n}):")
    for key, n in most_common:
        print(f"  {n:2d}x  {titles[key]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True,
                         help="Path to min_4stars_ai_music/ (contains ajam/hijaz/kurd/nahawand subfolders).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-per-song", type=int, default=None,
                         help="Cap how many surviving takes of the same poem go into the dataset. "
                              "Default: no cap (keep every 4-5* take).")
    parser.add_argument("--include-mood", action="store_true")
    parser.add_argument("--keep-control-header", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tracks = walk_corpus(args.dataset_root)
    print(f"\nResolved {len(tracks)} kept (4-5*) tracks across "
          f"{len(set(t.group_key for t in tracks))} distinct poems.")

    for maqam in MAQAMS.values():
        n = sum(1 for t in tracks if t.maqam == maqam)
        n_songs = len(set(t.group_key for t in tracks if t.maqam == maqam))
        print(f"  {maqam:10s} tracks={n:3d}  poems={n_songs:3d}")

    print_top_repeated_songs(tracks)

    tracks = cap_per_song(tracks, args.max_per_song)
    if args.max_per_song is not None:
        print(f"\nAfter capping at {args.max_per_song} takes/poem: {len(tracks)} tracks.")

    render_captions(tracks, include_mood=args.include_mood, keep_header=args.keep_control_header)

    train, val = split_train_val(tracks, args.val_fraction, args.seed)
    print(f"\nSplit: {len(train)} train / {len(val)} val tracks "
          f"({len(set(t.group_key for t in train))} / {len(set(t.group_key for t in val))} poems).")

    train_rows = write_split(train, args.out_dir / "train", args.dry_run)
    val_rows = write_split(val, args.out_dir / "val", args.dry_run)

    manifest_path = args.out_dir / "manifest.csv"
    all_rows = train_rows + val_rows
    if not all_rows:
        print("[warn] no tracks matched — check --dataset-root path.")
        return

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) + ["split"])
            writer.writeheader()
            for row in train_rows:
                writer.writerow({**row, "split": "train"})
            for row in val_rows:
                writer.writerow({**row, "split": "val"})
        print(f"\nWrote {manifest_path}")
    else:
        print(f"\n[dry-run] Would write {manifest_path} and copy {len(all_rows)} audio files.")


if __name__ == "__main__":
    main()
