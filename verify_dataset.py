#!/usr/bin/env python3
"""
verify_dataset.py — post-build QC for the YuE2 LoRA dataset.

Deliberately does NOT import prepare_dataset.py or maqam_prompt_generator.py.
It re-walks the corpus independently, so a bug in the builder can't hide by
being reproduced in the checker. Stdlib only.

    # check the built dataset on its own
    python verify_dataset.py --out-dir ./dataset

    # also diff it against the live corpus (catches tracks you deleted or
    # added since the build)
    python verify_dataset.py --out-dir ./dataset \
        --dataset-root ./min_4stars_ai_music

    # add ffprobe-based audio stats (needs ffmpeg installed)
    python verify_dataset.py --out-dir ./dataset --check-audio

Exit code is 0 if clean, 1 if any ERROR was reported. WARNs don't fail.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}
MAQAMS = {"hijaz", "nahawand", "ajam", "kurd"}

# Filenames written by prepare_dataset.py:
#   <maqam>_<poem_id:04d>_<slug>_take<NN>.<ext>
# The older, pre-bugfix scheme had no poem_id; detecting it tells the user
# their build predates the collision fix.
NAME_RE = re.compile(r"^(?P<maqam>[a-z]+)_(?P<poem>\d{4})_(?P<slug>.*)_take(?P<take>\d{2})$")
OLD_NAME_RE = re.compile(r"^(?P<maqam>[a-z]+)_(?P<slug>.*)_take(?P<take>\d{2})$")


class Report:
    def __init__(self) -> None:
        self.errors = 0
        self.warns = 0

    def error(self, msg: str) -> None:
        self.errors += 1
        print(f"  [ERROR] {msg}")

    def warn(self, msg: str) -> None:
        self.warns += 1
        print(f"  [warn]  {msg}")

    def ok(self, msg: str) -> None:
        print(f"  [ok]    {msg}")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_title_key(title: str) -> str:
    return re.sub(r"\s+", " ", nfc(title).strip())


def head(items, n: int = 8):
    """Print at most n examples so a systemic failure doesn't dump 250 lines."""
    items = list(items)
    for x in items[:n]:
        print(f"            {x}")
    if len(items) > n:
        print(f"            ... and {len(items) - n} more")


# --------------------------------------------------------------------------
# 1. Structure and pairing
# --------------------------------------------------------------------------

def _stem_and_kind(p: Path) -> tuple[str, str] | None:
    """Classify a file by its double extension. '<base>.style.txt' and
    '<base>.lyrics.txt' both end in .txt, so Path.stem alone isn't enough --
    peel one suffix at a time."""
    name = p.name
    if name.endswith(".style.txt"):
        return name[: -len(".style.txt")], "style"
    if name.endswith(".lyrics.txt"):
        return name[: -len(".lyrics.txt")], "lyrics"
    if p.suffix.lower() in AUDIO_EXTS:
        return p.stem, "audio"
    return None


def scan_split(split_dir: Path, rep: Report) -> dict[str, dict]:
    """Return {basename: {'audio': Path|None, 'style': Path|None, 'lyrics': Path|None}}."""
    entries: dict[str, dict] = defaultdict(lambda: {"audio": None, "style": None, "lyrics": None})
    for p in sorted(split_dir.iterdir()):
        if not p.is_file():
            continue
        classified = _stem_and_kind(p)
        if classified is None:
            rep.warn(f"unexpected file in {split_dir.name}/: {p.name}")
            continue
        stem, kind = classified
        if entries[stem][kind] is not None:
            rep.error(f"two {kind} files share the stem '{stem}' in {split_dir.name}/")
        entries[stem][kind] = p
    return dict(entries)


def check_pairing(split: str, entries: dict[str, dict], rep: Report) -> None:
    missing_style = [k for k, v in entries.items() if v["audio"] and not v["style"]]
    missing_lyrics = [k for k, v in entries.items() if v["audio"] and not v["lyrics"]]
    orphan_style = [k for k, v in entries.items() if v["style"] and not v["audio"]]
    orphan_lyrics = [k for k, v in entries.items() if v["lyrics"] and not v["audio"]]
    empty_audio = [k for k, v in entries.items()
                   if v["audio"] and v["audio"].stat().st_size == 0]
    empty_style = [k for k, v in entries.items()
                   if v["style"] and not v["style"].read_text(encoding="utf-8").strip()]
    # An empty lyrics file is a WARN, not an ERROR: instrumental tracks can
    # legitimately have no lyrics. It's only worth flagging, not failing.
    empty_lyrics = [k for k, v in entries.items()
                    if v["lyrics"] and not v["lyrics"].read_text(encoding="utf-8").strip()]

    if missing_style:
        rep.error(f"{split}: {len(missing_style)} audio files have no .style.txt")
        head(missing_style)
    if missing_lyrics:
        rep.error(f"{split}: {len(missing_lyrics)} audio files have no .lyrics.txt")
        head(missing_lyrics)
    if orphan_style:
        rep.error(f"{split}: {len(orphan_style)} .style.txt files have no audio")
        head(orphan_style)
    if orphan_lyrics:
        rep.error(f"{split}: {len(orphan_lyrics)} .lyrics.txt files have no audio")
        head(orphan_lyrics)
    if empty_audio:
        rep.error(f"{split}: {len(empty_audio)} zero-byte audio files")
        head(empty_audio)
    if empty_style:
        rep.error(f"{split}: {len(empty_style)} empty .style.txt files")
        head(empty_style)
    if empty_lyrics:
        rep.warn(f"{split}: {len(empty_lyrics)} tracks have empty lyrics "
                  f"(fine if genuinely instrumental, otherwise check the source)")
        head(empty_lyrics)
    if not (missing_style or missing_lyrics or orphan_style or orphan_lyrics
            or empty_audio or empty_style):
        rep.ok(f"{split}: {len(entries)} audio/style/lyrics triples, all present and non-empty")


def check_naming(split: str, entries: dict[str, dict], rep: Report) -> None:
    """Catch builds made before the poem_id collision fix."""
    old_scheme = 0
    unparseable = []
    for stem in entries:
        if NAME_RE.match(stem):
            continue
        if OLD_NAME_RE.match(stem):
            old_scheme += 1
        else:
            unparseable.append(stem)
    if old_scheme:
        rep.error(
            f"{split}: {old_scheme} files use the OLD naming scheme "
            f"(<maqam>_<slug>_take<NN>, no poem id). This build predates the "
            f"filename-collision fix in prepare_dataset.py -- poems may have "
            f"silently overwritten each other. Rebuild from scratch."
        )
    if unparseable:
        rep.warn(f"{split}: {len(unparseable)} filenames don't match either naming scheme")
        head(unparseable)


# --------------------------------------------------------------------------
# 2. Captions
# --------------------------------------------------------------------------

def check_captions(split: str, entries: dict[str, dict], rep: Report) -> None:
    by_text: dict[str, int] = Counter()
    maqam_disagree = []
    for stem, v in entries.items():
        if not v["style"]:
            continue
        text = v["style"].read_text(encoding="utf-8").strip()
        by_text[text] += 1

        m = NAME_RE.match(stem) or OLD_NAME_RE.match(stem)
        if not m:
            continue
        file_maqam = m.group("maqam")
        found = re.search(r"Maqam\s+([A-Za-z]+)", text)
        if found and found.group(1).lower() != file_maqam:
            maqam_disagree.append(f"{stem}: filename={file_maqam} caption={found.group(1)}")
        elif not found:
            maqam_disagree.append(f"{stem}: caption names no maqam at all")

    if maqam_disagree:
        rep.error(f"{split}: {len(maqam_disagree)} captions disagree with the filename's maqam")
        head(maqam_disagree)
    else:
        rep.ok(f"{split}: every caption names the maqam its filename claims")

    print(f"            {len(by_text)} distinct caption texts across {sum(by_text.values())} files")
    if len(by_text) > len(MAQAMS) * 3:
        rep.warn(
            f"{split}: {len(by_text)} distinct captions is high for a closed template. "
            f"Expect ~4 (one per maqam) without --include-mood. Check for template drift."
        )


def check_lyrics(split: str, entries: dict[str, dict], rep: Report) -> None:
    marker_leaked = []
    pipe_tag_leaked = []
    for stem, v in entries.items():
        if not v["lyrics"]:
            continue
        text = v["lyrics"].read_text(encoding="utf-8")
        for line in text.splitlines():
            s = line.strip()
            if s and set(s) <= set("/*"):
                marker_leaked.append(stem)
                break
        if re.search(r"\[[^\]]*\|[^\]]*\]", text):
            pipe_tag_leaked.append(stem)

    if marker_leaked:
        rep.error(f"{split}: {len(marker_leaked)} lyrics files still contain a Suno UI "
                  f"marker line (e.g. '///***///') that should have been stripped")
        head(marker_leaked)
    if pipe_tag_leaked:
        rep.warn(f"{split}: {len(pipe_tag_leaked)} lyrics files still contain a "
                 f"pipe-delimited production tag (e.g. '[Verse 1 | ...]') -- expected "
                 f"only if --lyrics-tag-mode full was used deliberately")
        head(pipe_tag_leaked)
    if not marker_leaked and not pipe_tag_leaked:
        rep.ok(f"{split}: lyrics sidecars are clean of Suno UI markers and raw production tags")


# --------------------------------------------------------------------------
# 3. Manifest cross-check
# --------------------------------------------------------------------------

def load_manifest(manifest_path: Path, rep: Report) -> list[dict]:
    with manifest_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        rep.error("manifest.csv is empty")
        return []
    required = {"clip_id", "original_title", "maqam", "dest_audio",
                "dest_style", "dest_lyrics", "split"}
    missing = required - set(rows[0].keys())
    if missing:
        rep.error(f"manifest.csv is missing columns: {sorted(missing)}")
    return rows


def check_manifest(rows: list[dict], out_dir: Path,
                   entries_by_split: dict[str, dict], rep: Report) -> None:
    # Duplicate destinations are the collision signature.
    dest_counts = Counter(r["dest_audio"] for r in rows)
    dupes = [d for d, n in dest_counts.items() if n > 1]
    if dupes:
        rep.error(
            f"{len(dupes)} destination paths appear more than once in manifest.csv -- "
            f"these tracks overwrote each other on copy"
        )
        head(dupes)
    else:
        rep.ok(f"manifest.csv: {len(rows)} rows, all destination paths unique")

    dupe_clips = [c for c, n in Counter(r["clip_id"] for r in rows).items() if n > 1]
    if dupe_clips:
        rep.error(f"{len(dupe_clips)} clip_ids appear more than once in manifest.csv")
        head(dupe_clips)

    # Every manifest row should exist on disk, and vice versa.
    on_disk = {
        str(v["audio"].resolve())
        for entries in entries_by_split.values()
        for v in entries.values() if v["audio"]
    }
    in_manifest = set()
    absent = []
    relocated = 0
    for r in rows:
        p = Path(r["dest_audio"])
        if not p.exists():
            # manifest.csv records the path as it was at build time. If the
            # dataset dir was moved/renamed, or the verifier is run from a
            # different cwd than the build was, fall back to matching by
            # basename inside the split we expect it in.
            alt = out_dir / r.get("split", "") / p.name
            if alt.exists():
                p, relocated = alt, relocated + 1
        in_manifest.add(str(p.resolve()))
        if not p.exists():
            absent.append(r["dest_audio"])
    if relocated:
        rep.warn(f"{relocated} manifest paths only resolved after relocating to {out_dir} "
                 f"(dataset dir was moved, or built from a different working directory)")

    if absent:
        rep.error(f"{len(absent)} manifest rows point at audio files that don't exist")
        head(absent)
    orphans = on_disk - in_manifest
    if orphans:
        rep.error(f"{len(orphans)} audio files on disk have no manifest row")
        head(sorted(orphans))
    if not absent and not orphans:
        rep.ok("manifest.csv and the files on disk agree exactly")


# --------------------------------------------------------------------------
# 4. Train/val leakage
# --------------------------------------------------------------------------

def check_leakage(rows: list[dict], rep: Report) -> None:
    groups: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        key = f"{r['maqam']}::{normalize_title_key(r['original_title'])}"
        groups[key].add(r["split"])

    leaked = sorted(k for k, splits in groups.items() if len(splits) > 1)
    if leaked:
        rep.error(
            f"{len(leaked)} poems appear in BOTH train and val -- validation loss "
            f"will be optimistic because the lyrics were memorized in training"
        )
        head(leaked)
    else:
        rep.ok(f"no poem spans train and val ({len(groups)} distinct poems)")

    per = defaultdict(lambda: {"train": 0, "val": 0})
    for r in rows:
        per[r["maqam"]][r["split"]] += 1
    print("\n  Tracks per maqam:")
    print(f"    {'maqam':12s} {'train':>6s} {'val':>5s} {'total':>6s} {'val %':>7s}")
    for maqam in sorted(per):
        t, v = per[maqam]["train"], per[maqam]["val"]
        pct = (100 * v / (t + v)) if (t + v) else 0
        print(f"    {maqam:12s} {t:6d} {v:5d} {t + v:6d} {pct:6.1f}%")
    empty = [m for m, c in per.items() if c["val"] == 0]
    if empty:
        rep.warn(f"maqams with no val tracks at all: {empty}")


# --------------------------------------------------------------------------
# 5. Drift vs the live corpus
# --------------------------------------------------------------------------

def walk_corpus(dataset_root: Path, rep: Report) -> dict[str, str]:
    """Independent re-walk. Returns {clip_id: 'maqam::title'} for survivors."""
    survivors: dict[str, str] = {}
    for maqam_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        maqam = maqam_dir.name.strip().lower()
        if maqam not in MAQAMS:
            rep.warn(f"skipping unrecognized top-level folder: {maqam_dir.name}")
            continue
        for manifest_path in sorted(maqam_dir.rglob("workspace_manifest.json")):
            ws = manifest_path.parent
            names_on_disk = {nfc(p.name) for p in ws.iterdir() if p.is_file()}
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                rep.error(f"unreadable manifest {manifest_path}: {e}")
                continue
            for rt in data.get("tracks", []):
                if nfc(rt["assigned_filename"]) in names_on_disk:
                    survivors[rt["clip_id"]] = f"{maqam}::{normalize_title_key(rt['original_title'])}"
    return survivors


def check_drift(rows: list[dict], dataset_root: Path, rep: Report) -> None:
    survivors = walk_corpus(dataset_root, rep)
    built = {r["clip_id"]: f"{r['maqam'].lower()}::{normalize_title_key(r['original_title'])}"
             for r in rows}

    stale = sorted(set(built) - set(survivors))
    missing = sorted(set(survivors) - set(built))

    print(f"\n  Corpus now has {len(survivors)} surviving tracks; "
          f"dataset was built from {len(built)}.")
    if stale:
        rep.error(
            f"{len(stale)} tracks are IN the dataset but no longer in the corpus "
            f"(deleted since the build) -- these are training on audio you rejected"
        )
        head(f"{c}  {built[c]}" for c in stale)
    if missing:
        rep.error(
            f"{len(missing)} tracks survive in the corpus but are NOT in the dataset "
            f"(added since the build, or dropped by a bug)"
        )
        head(f"{c}  {survivors[c]}" for c in missing)
    if not stale and not missing:
        rep.ok("dataset is in sync with the corpus, clip-for-clip")


# --------------------------------------------------------------------------
# 6. Optional audio stats (bridge to the loudness/silence question)
# --------------------------------------------------------------------------

def check_audio(entries_by_split: dict[str, dict], rep: Report) -> None:
    if not shutil.which("ffprobe"):
        rep.warn("--check-audio requested but ffprobe isn't on PATH; skipping")
        return
    durations, rates, channels, codecs = [], Counter(), Counter(), Counter()
    failed = []
    paths = [v["audio"] for entries in entries_by_split.values()
             for v in entries.values() if v["audio"]]
    for p in paths:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=sample_rate,channels,codec_name:format=duration",
                 "-of", "json", str(p)],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout
            info = json.loads(out)
            stream = (info.get("streams") or [{}])[0]
            rates[stream.get("sample_rate", "?")] += 1
            channels[stream.get("channels", "?")] += 1
            codecs[stream.get("codec_name", "?")] += 1
            durations.append(float(info["format"]["duration"]))
        except Exception as e:
            failed.append(f"{p.name}: {type(e).__name__}")

    if failed:
        rep.error(f"{len(failed)} files could not be probed (likely corrupt)")
        head(failed)
    if not durations:
        return

    durations.sort()
    total = sum(durations)
    print(f"\n  Audio: {len(durations)} files, {total / 3600:.2f} h total")
    print(f"    duration  min {durations[0]:.1f}s  median "
          f"{durations[len(durations) // 2]:.1f}s  max {durations[-1]:.1f}s")
    print(f"    sample rates: {dict(rates)}")
    print(f"    channels:     {dict(channels)}")
    print(f"    codecs:       {dict(codecs)}")
    if len(rates) > 1 or len(channels) > 1 or len(codecs) > 1:
        rep.warn("mixed sample rates/channels/codecs -- standardize before training")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="The built dataset dir (contains train/, val/, manifest.csv).")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="min_4stars_ai_music/ -- enables the drift check against the live corpus.")
    ap.add_argument("--check-audio", action="store_true",
                    help="Probe every file with ffprobe for duration/rate/channels.")
    args = ap.parse_args()

    rep = Report()

    print("=" * 70)
    print("1. Structure")
    print("=" * 70)
    if not args.out_dir.is_dir():
        print(f"  [ERROR] {args.out_dir} does not exist -- the build was never run.")
        return 1

    entries_by_split: dict[str, dict] = {}
    for split in ("train", "val"):
        d = args.out_dir / split
        if not d.is_dir():
            rep.error(f"{args.out_dir}/{split}/ is missing")
            continue
        entries_by_split[split] = scan_split(d, rep)
    if not entries_by_split:
        return 1
    rep.ok(f"found {'/'.join(entries_by_split)} under {args.out_dir}")

    print("\n" + "=" * 70)
    print("2. Pairing and naming")
    print("=" * 70)
    for split, entries in entries_by_split.items():
        check_pairing(split, entries, rep)
        check_naming(split, entries, rep)

    print("\n" + "=" * 70)
    print("3. Captions")
    print("=" * 70)
    for split, entries in entries_by_split.items():
        check_captions(split, entries, rep)
        check_lyrics(split, entries, rep)

    manifest_path = args.out_dir / "manifest.csv"
    rows: list[dict] = []
    print("\n" + "=" * 70)
    print("4. Manifest cross-check")
    print("=" * 70)
    if not manifest_path.exists():
        rep.error("manifest.csv is missing -- can't cross-check or test for leakage")
    else:
        rows = load_manifest(manifest_path, rep)
        if rows:
            check_manifest(rows, args.out_dir, entries_by_split, rep)

    if rows:
        print("\n" + "=" * 70)
        print("5. Train/val leakage")
        print("=" * 70)
        check_leakage(rows, rep)

    if args.dataset_root:
        print("\n" + "=" * 70)
        print("6. Drift vs the live corpus")
        print("=" * 70)
        if not args.dataset_root.is_dir():
            rep.error(f"--dataset-root {args.dataset_root} does not exist")
        elif not rows:
            rep.error("can't check drift without a readable manifest.csv")
        else:
            check_drift(rows, args.dataset_root, rep)

    if args.check_audio:
        print("\n" + "=" * 70)
        print("7. Audio stats")
        print("=" * 70)
        check_audio(entries_by_split, rep)

    print("\n" + "=" * 70)
    print(f"RESULT: {rep.errors} error(s), {rep.warns} warning(s)")
    print("=" * 70)
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
