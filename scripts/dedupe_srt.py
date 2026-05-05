"""Strip an SRT and dedupe YouTube's rolling auto-captions.

YouTube's auto-generated captions ship as overlapping rolling text:
each new line in the SRT repeats the previous line's tail plus a
few new words. Naive concatenation produces transcripts that read
the same sentence 2–3 times. This dedupes against the most recent
emitted line, also dropping wholly-contained substrings.

Usage::

    python dedupe_srt.py path/to/video.en.srt > transcript.txt

Or pipe in a file via stdin::

    cat video.en.srt | python dedupe_srt.py > transcript.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def dedupe_srt(srt_text: str) -> str:
    """Return cleaned transcript text from raw SRT contents."""
    seen: list[str] = []
    for block in srt_text.strip().split("\n\n"):
        parts = block.split("\n")
        if len(parts) < 3:
            continue
        # SRT block format:
        #   1. index
        #   2. HH:MM:SS,mmm --> HH:MM:SS,mmm
        #   3+. caption text (may span multiple lines)
        line = " ".join(parts[2:]).strip()
        if not line:
            continue
        if not seen:
            seen.append(line)
            continue
        prev = seen[-1]
        if line == prev:
            continue
        # Substring overlap — keep whichever is longer
        if line in prev:
            continue
        if prev in line:
            seen[-1] = line
            continue
        seen.append(line)
    text = " ".join(seen)
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dedupe YouTube auto-caption SRT into clean text."
    )
    p.add_argument(
        "srt",
        nargs="?",
        type=Path,
        help="Path to .srt file (omit to read from stdin).",
    )
    args = p.parse_args()
    raw = args.srt.read_text() if args.srt else sys.stdin.read()
    sys.stdout.write(dedupe_srt(raw))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
