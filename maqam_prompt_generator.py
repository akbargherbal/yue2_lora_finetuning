#!/usr/bin/env python3
"""
Maqam Prompt Generator (v2 — "Winning Template" edition)
----------------------------------------------------------
Locks in the exact prompt structure that won after 2.5 months of manual
trial-and-error (Suno v5, cinematic orchestral rock, Arabic male vocal).

Only the Maqam name changes between generations. Every other field stays
fixed on purpose: an LLM (Gemini) was previously asked to vary a `mood`
field per-maqam ("happy" for one, "sad" for another) and introduced
unwanted drift. That field has been removed entirely rather than fixed,
since it was never essential to the winning prompt.

Only the four widely-known maqams with clean genre/mood behavior are
offered: Hijaz, Nahawand, Ajam, Kurd. (Rast, Bayati, Sikah, Saba, etc.
rely on quarter-tone intervals with no faithful Western mapping and were
excluded from the original tool for the same reason.)

Two track-type modes are supported:
  - standard      (default, unless overridden via CLI flag): everything as
                   before — asks for a start phrase/verse, includes the
                   "vocals" field, includes [START_ON: ...] in the header.
  - instrumental: for backing tracks — no lyrics, so no start phrase and
                   no "vocals" field are included. This is NOT the same as
                   excluding vocalization: non-lyrical/wordless vocals are
                   still left possible on purpose, so nothing is added to
                   EXCLUDE for this mode either. That choice is left to Suno.

CLI usage:
    python maqam_prompt_generator.py                 # asks for mode interactively
    python maqam_prompt_generator.py --standard       # skips the mode prompt
    python maqam_prompt_generator.py --instrumental    # skips the mode prompt
"""

import argparse
import os
import sys

if sys.platform == "win32":
    os.system("")


class C:
    _enabled = sys.stdout.isatty()
    RESET = "\033[0m" if _enabled else ""
    BOLD = "\033[1m" if _enabled else ""
    DIM = "\033[2m" if _enabled else ""
    CYAN = "\033[36m" if _enabled else ""
    YELLOW = "\033[33m" if _enabled else ""
    GREEN = "\033[32m" if _enabled else ""
    MAGENTA = "\033[35m" if _enabled else ""


# Menu order matches how the maqams were requested: Hijaz, Nahawand, Ajam, Kurd
MAQAMS = {
    1: "Hijaz",
    2: "Nahawand",
    3: "Ajam",
    4: "Kurd",
}

# grand concert hall or epic stadium acoustics
# Fixed fields — identical across every maqam. Only "vocals" (standard mode
# only), "genre" (instrumental mode prefixes "Full Instrumental"), and
# "production" (wording differs slightly by mode) change.
GENRE_STANDARD = (
    "Symphonic cinematic orchestral ballad, hymn-like grand concert hall acoustics, "
    "heavy rock instrumentation, stately groove, 110 BPM."
)
# "Full Instrumental" goes first in the field, as an explicit signal to Suno
# that there's no lead vocal track — separate from the EXCLUDE list, which
# intentionally says nothing about vocalization either way (see note below).
GENRE_INSTRUMENTAL = (
    "Full Instrumental, "
    "Symphonic cinematic orchestral ballad, hymn-like grand concert hall acoustics, "
    "heavy rock instrumentation, stately groove, 110 BPM."
)

# PRODUCTION_STANDARD assumes lyrical vocals are present and describes the
# mix ducking around them ("forward vocals pulling instrumentation down...").
# PRODUCTION_INSTRUMENTAL keeps the exact same mix dynamic but generalizes
# "vocals" to "lead line" so it stays accurate whether Suno adds a lead
# instrument, non-lyrical vocalization, or nothing there at all — the
# instrumental mode never presumes vocals but also never rules them out.
PRODUCTION_STANDARD = (
    "Audiophile recording, punchy centered mix, forward vocals pulling "
    "instrumentation down on sustained phrases then band re-enters between "
    "lines, bright presence, clean transients, large dynamic range, natural "
    "breath room between phrases."
)
PRODUCTION_INSTRUMENTAL = (
    "Audiophile recording, punchy centered mix, forward lead line pulling "
    "instrumentation down on sustained phrases then band re-enters between "
    "lines, bright presence, clean transients, large dynamic range, natural "
    "breath room between phrases."
)

INSTRUMENTATION = (
    "Distorted electric guitars, orchestral strings, weighted acoustic rock "
    "drums, tight rhythm section."
)
# NOTE: EXCLUDE is identical for both "standard" and "instrumental" modes.
# Instrumental/backing-track mode does NOT add "no vocals" / "no vocalization"
# here on purpose — an instrumental track can still legitimately have
# non-lyrical vocalization (humming, ad-lib vocalizing, wordless vocals),
# so that decision is left entirely to Suno rather than forced either way.
EXCLUDE = (
    "Oud, Qanun, Darbuka, Tabla, Ney, Buzuq, Sitar, Khaliji, female vocals, fast tempo, "
    "upbeat, speed metal, punk, muddy mix, muffled vocals, distant vocals, "
    "washed out, wall of sound, synth pads, extreme "
    "panning, autotune, vocal strain, growling, audience, applause"
)

# The two generation modes.
#   standard     -> current behavior: asks for a start phrase/verse, includes
#                   the "vocals" field (Arabic lyrical diction/melisma) and a
#                   [START_ON: ...] header.
#   instrumental -> backing-track mode: no lyrics, so no start phrase and no
#                   "vocals" field is emitted. This says nothing about
#                   whether vocalization occurs — that's left to Suno (see
#                   EXCLUDE note above).
MODES = ("standard", "instrumental")


def build_prefix(start_phrase: str | None) -> str:
    """Fixed MAX-mode header, plus the opening phrase/verse when present.

    In instrumental mode there is no lyric to start on, so `start_phrase`
    is None/empty and the [START_ON: ...] lines are omitted entirely.
    """
    header = "[Is_MAX_MODE: MAX](MAX) [QUALITY: MAX](MAX) [REALISM: MAX](MAX)"
    if start_phrase and start_phrase.strip():
        return header + "\n[START_ON: TRUE]\n" + f'[START_ON: "{start_phrase}"]'
    return header


RULE_WIDTH = 64


def rule(char="─", color=C.DIM):
    print(f"{color}{char * RULE_WIDTH}{C.RESET}")


def print_menu():
    print()
    rule("═", C.CYAN)
    print(f"{C.BOLD}  MAQAM PROMPT GENERATOR{C.RESET}")
    rule("═", C.CYAN)
    for num, name in MAQAMS.items():
        print(f"  {C.YELLOW}{num}.{C.RESET} {C.BOLD}Maqam {name}{C.RESET}")
    print()
    rule()


def get_mode() -> str:
    """Interactive standard-vs-instrumental prompt. Only used when the mode
    wasn't already supplied via --standard/--instrumental on the CLI."""
    print()
    rule("═", C.CYAN)
    print(f"{C.BOLD}  TRACK TYPE{C.RESET}")
    rule("═", C.CYAN)
    print(
        f"  {C.YELLOW}1.{C.RESET} {C.BOLD}Standard{C.RESET} (lyrics + start phrase + vocals field)"
    )
    print(
        f"  {C.YELLOW}2.{C.RESET} {C.BOLD}Instrumental{C.RESET} (backing track, no lyrics/start phrase)"
    )
    print()
    rule()
    while True:
        raw = input(f"\n{C.CYAN}➤ Enter a number (1-2):{C.RESET} ").strip()
        if raw == "1":
            return "standard"
        if raw == "2":
            return "instrumental"
        print(f"{C.MAGENTA}  Invalid choice, please try again.{C.RESET}")


def get_choice():
    while True:
        raw = input(f"\n{C.CYAN}➤ Enter a number (1-{len(MAQAMS)}):{C.RESET} ").strip()
        if raw.isdigit() and int(raw) in MAQAMS:
            return int(raw)
        print(f"{C.MAGENTA}  Invalid choice, please try again.{C.RESET}")


def get_start_phrase():
    while True:
        raw = input(
            f"\n{C.CYAN}➤ Enter the phrase/verse to start on:{C.RESET} "
        ).strip()
        if raw:
            return raw
        print(f"{C.MAGENTA}  Phrase cannot be empty, please try again.{C.RESET}")


def get_mood():
    """Optional — blank/whitespace-only input means "no mood field"."""
    raw = input(
        f"\n{C.CYAN}➤ Enter a mood (optional, press Enter to skip):{C.RESET} "
    ).strip()
    return raw or None


def build_vocals(maqam_name: str) -> str:
    return (
        "deep male vocals, mixed-voice chest-head resonance blend on "
        "sustained notes, breath-supported melismatic runs, controlled "
        "vibrato, full-voiced commanding presence, precise Arabic "
        f"diction, melismatic phrasing in Maqam {maqam_name} with unhurried "
        "phrase-ending sustains."
    )


def build_prompt(
    maqam_name: str,
    start_phrase: str | None,
    mood: str | None = None,
    instrumental: bool = False,
) -> str:
    """Returns the full markdown block: PREFIX + PROMPT (genre [-> vocals] -> production -> instrumentation [-> mood]) + EXCLUDE.

    `mood` is optional. If None, empty, or whitespace-only, the mood field
    is omitted entirely from the prompt block rather than emitted empty.

    `instrumental` controls two things:
      - the "vocals" field is left out of the prompt block entirely
      - `start_phrase` is ignored (no lyric to start on), so the
        [START_ON: ...] header lines are omitted too.
    This does NOT add anything to EXCLUDE — instrumental tracks may still
    contain non-lyrical vocalization; that's left to Suno's discretion.
    """
    genre = GENRE_INSTRUMENTAL if instrumental else GENRE_STANDARD
    prompt_block = f'genre: "{genre}"'

    if not instrumental:
        vocals = build_vocals(maqam_name)
        prompt_block += f'\nvocals: "{vocals}"'

    production = PRODUCTION_INSTRUMENTAL if instrumental else PRODUCTION_STANDARD
    prompt_block += (
        f'\nproduction: "{production}"\ninstrumentation: "{INSTRUMENTATION}"'
    )

    if mood and mood.strip():
        prompt_block += f'\nmood: "{mood.strip()}"'

    prefix = build_prefix(None if instrumental else start_phrase)

    return (
        "PROMPT:\n\n"
        "```\n"
        f"{prefix}\n\n{prompt_block}\n"
        "```\n\n"
        "EXCLUDE:\n\n"
        "```\n"
        f"{EXCLUDE}\n"
        "```"
    )


def generate_prompt(
    choice,
    start_phrase: str | None = None,
    mood: str | None = None,
    instrumental: bool = False,
) -> str:
    """
    Generate a prompt without the interactive menu.

    Parameters
    ----------
    choice : int | str
        Either a menu number (e.g. 2) or a maqam name (e.g. "Nahawand").
    start_phrase : str | None, optional
        The phrase/verse the vocal should start on, e.g. "آذَنَتْنَا بِبَيْنِهَا".
        Required (non-empty) when `instrumental` is False. Ignored when
        `instrumental` is True, since instrumental/backing tracks have no
        lyric to start on.
    mood : str | None, optional
        Optional mood field appended to the end of the positive prompt.
        Omitted entirely if None, empty, or whitespace-only.
    instrumental : bool, optional
        If True, generates a backing-track prompt: no "vocals" field and
        no [START_ON: ...] header. This does NOT exclude vocalization —
        non-lyrical vocals are still left possible; that call is left to
        Suno rather than forced via EXCLUDE.

    Examples
    --------
    >>> generate_prompt(2, "آذَنَتْنَا بِبَيْنِهَا")
    >>> generate_prompt("Nahawand", "آذَنَتْنَا بِبَيْنِهَا")
    >>> generate_prompt("Nahawand", "آذَنَتْنَا بِبَيْنِهَا", mood="wistful, elegiac")
    >>> generate_prompt("Nahawand", instrumental=True)
    >>> generate_prompt("Nahawand", instrumental=True, mood="wistful, elegiac")
    """
    if not instrumental:
        if not isinstance(start_phrase, str) or not start_phrase.strip():
            raise ValueError(
                "start_phrase must be a non-empty string when instrumental=False."
            )

    if mood is not None and not isinstance(mood, str):
        raise TypeError("mood must be a string or None.")

    if isinstance(choice, int):
        if choice not in MAQAMS:
            raise ValueError(f"Choice must be between 1 and {len(MAQAMS)}.")
        return build_prompt(MAQAMS[choice], start_phrase, mood, instrumental)

    if isinstance(choice, str):
        choice_norm = choice.strip().lower()
        for name in MAQAMS.values():
            if name.lower() == choice_norm:
                return build_prompt(name, start_phrase, mood, instrumental)
        valid = ", ".join(MAQAMS.values())
        raise ValueError(f"Unknown maqam '{choice}'. Valid maqams: {valid}")

    raise TypeError("choice must be an int or a str")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Maqam Prompt Generator — standard (lyrical) or instrumental (backing track)."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--standard",
        action="store_true",
        help="Skip the mode prompt and generate a standard (lyrical) prompt.",
    )
    mode_group.add_argument(
        "--instrumental",
        action="store_true",
        help="Skip the mode prompt and generate an instrumental/backing-track prompt "
        "(no lyrics, no start phrase, no vocals field).",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    if args.standard:
        mode = "standard"
    elif args.instrumental:
        mode = "instrumental"
    else:
        mode = get_mode()

    instrumental = mode == "instrumental"

    print_menu()
    choice = get_choice()
    maqam_name = MAQAMS[choice]

    start_phrase = None if instrumental else get_start_phrase()
    mood = get_mood()

    print()
    rule("═", C.GREEN)
    print(f"{C.GREEN}✔ Mode:{C.RESET}     {C.BOLD}{mode.capitalize()}{C.RESET}")
    print(f"{C.GREEN}✔ Selected:{C.RESET} {C.BOLD}Maqam {maqam_name}{C.RESET}")
    if not instrumental:
        print(f"{C.GREEN}✔ Start on:{C.RESET} {C.BOLD}{start_phrase}{C.RESET}")
    if mood:
        print(f"{C.GREEN}✔ Mood:{C.RESET}    {C.BOLD}{mood}{C.RESET}")
    rule("═", C.GREEN)
    print()
    print(build_prompt(maqam_name, start_phrase, mood, instrumental))
    print()


if __name__ == "__main__":
    main()
