#!/usr/bin/env python3
"""Mirror the high-stake run artifacts to GCS on a timer.

The YuE2 trainer has no time-based save -- `--save-every` is in optimizer steps --
so a Colab/VM interruption can lose GPU work. Run this in a second terminal next
to training: it mirrors only the folders needed to resume or debug the run up to
a GCS prefix with `gsutil rsync`.

rsync is append/update-only here (no `-d`): nothing is ever deleted on the remote
side. If a pass catches a checkpoint mid-write, the next pass re-uploads it once
the local file stops changing, so a torn upload self-heals.

Usage:
    python backup_to_gcp.py                     # every 25 min, first pass at start
    python backup_to_gcp.py --interval-minutes 20
    python backup_to_gcp.py --once              # a single pass
    python backup_to_gcp.py --dry-run           # show what would sync, upload nothing
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = "gs://akbar-december-2024-backup/YuE2-3B_13092026/run_backup"
DEFAULT_LOG = Path("/content/logs/gcp_backup.log")

# (source folder, remote subfolder, wait for writes to settle before syncing)
TARGETS = [
    (REPO_ROOT / "ComfyUI" / "models" / "loras", "loras", True),
    (Path("/content/logs"), "logs", False),
    (REPO_ROOT / "agent_notes", "agent_notes", False),
]
CACHE_DIR = REPO_ROOT / "ComfyUI" / "custom_nodes" / "ComfyUI-YuE2-Trainer" / "cache"
DEFAULT_EXCLUDES = [r".*\.tmp$", r".*put_loras_here$", r".*put_checkpoints_here$"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=DEFAULT_BASE, help=f"GCS prefix to mirror into (default: {DEFAULT_BASE})")
    p.add_argument("--interval-minutes", type=float, default=25.0, help="Minutes between passes (default: 25)")
    p.add_argument("--once", action="store_true", help="Run one pass and exit (for cron).")
    p.add_argument("--dry-run", action="store_true", help="Log the sync commands but upload nothing.")
    p.add_argument("--include-cache", action="store_true",
                   help="Also mirror the VAE latent cache (large; regenerable, but saves the ~20 min encode).")
    p.add_argument("--settle-seconds", type=float, default=60.0,
                   help="For checkpoint folders, wait until the newest file is this old before syncing (default: 60).")
    p.add_argument("--gsutil", default="gsutil", help="gsutil executable to use.")
    p.add_argument("--log-file", default=str(DEFAULT_LOG), help=f"Log file (default: {DEFAULT_LOG})")
    p.add_argument("--exclude", action="append", default=[], help="Extra gsutil -x regex to exclude (repeatable).")
    return p.parse_args()


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("gcp_backup")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def wait_for_settle(folder: Path, seconds: float, logger: logging.Logger) -> None:
    """Block until the newest file in `folder` has been untouched for `seconds`."""
    if seconds <= 0:
        return
    while True:
        newest = max((p.stat().st_mtime for p in folder.rglob("*") if p.is_file()), default=0.0)
        age = time.time() - newest
        if age >= seconds:
            return
        wait = min(seconds - age, 5.0)
        logger.info("%s: newest file is %.0fs old, waiting %.0fs for the write to finish",
                    folder.name, age, wait)
        time.sleep(wait)


def sync(src: Path, dst: str, args: argparse.Namespace, logger: logging.Logger) -> bool:
    # gsutil honours only the last -x flag, so combine every pattern into one alternation.
    patterns = DEFAULT_EXCLUDES + args.exclude
    cmd = [args.gsutil, "-m", "rsync", "-r", "-x", "(" + "|".join(patterns) + ")", str(src), dst.rstrip("/") + "/"]
    logger.info("syncing %s -> %s", src, dst)
    if args.dry_run:
        logger.info("dry-run: %s", " ".join(cmd))
        return True
    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("sync failed (exit %d): %s", result.returncode, (result.stderr or result.stdout).strip()[-2000:])
        return False
    logger.info("ok: %s in %.1fs", src, time.perf_counter() - started)
    return True


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)

    gsutil = shutil.which(args.gsutil)
    if not gsutil:
        logger.error("could not find '%s' on PATH", args.gsutil)
        return 2
    args.gsutil = gsutil

    targets = list(TARGETS)
    if args.include_cache:
        targets.append((CACHE_DIR, "latent_cache", False))

    logger.info("backup to %s every %.0f min (once=%s, dry-run=%s)", args.base, args.interval_minutes,
                args.once, args.dry_run)
    for src, sub, _ in targets:
        logger.info("  watching %s -> %s/%s", src, args.base.rstrip("/"), sub)

    try:
        while True:
            failures = 0
            for src, sub, settle in targets:
                if not src.is_dir():
                    logger.warning("skipping missing folder %s", src)
                    continue
                if settle:
                    wait_for_settle(src, args.settle_seconds, logger)
                if not sync(src, f"{args.base.rstrip('/')}/{sub}", args, logger):
                    failures += 1
            logger.info("pass complete: %d/%d folders synced", len(targets) - failures, len(targets))
            if args.once:
                break
            time.sleep(max(1.0, args.interval_minutes * 60.0))
    except KeyboardInterrupt:
        logger.info("interrupted; last completed pass is what is in GCS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
