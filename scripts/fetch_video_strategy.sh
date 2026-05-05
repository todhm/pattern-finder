#!/usr/bin/env bash
#
# fetch_video_strategy.sh — pull a YouTube trading-strategy video for offline study.
#
# Downloads the mp4 (480p, capped to keep the file ~20–30 MB), the auto-
# generated English subtitles, and produces a deduped transcript.txt
# suitable for reading by a human or pasting into a note-taking tool.
#
# Run inside the backtester container so the deps (yt-dlp + ffmpeg)
# are available without polluting the host:
#
#     docker compose exec backtester bash scripts/fetch_video_strategy.sh \
#         "https://www.youtube.com/watch?v=W5Hxv3hL3vY" /tmp/orb
#
# First positional arg = YouTube URL. Second = output dir (default /tmp/yt-strategy).
#
# Outputs
#   <out>/video.mp4          — for ad-hoc frame inspection
#   <out>/video.en.srt       — raw auto-captions
#   <out>/transcript.txt     — deduped one-paragraph transcript
#   <out>/frames/*.png       — keyframes (~1 every 30s) for visual study
#
# Why this exists: webpage scrapers can't see YouTube transcripts, so for
# strategy research we download the artifact end-to-end and dedupe locally.
# The frame extractor lets us read chart screenshots when the spoken rules
# are ambiguous ("the rejection candle here" — pull the frame at that
# timestamp).

set -euo pipefail

URL="${1:-}"
OUT="${2:-/tmp/yt-strategy}"

if [[ -z "$URL" ]]; then
    echo "Usage: $0 <youtube_url> [out_dir]" >&2
    exit 2
fi

# Ensure deps. yt-dlp is pip-installable; ffmpeg comes from apt.
need() { command -v "$1" >/dev/null || { echo "Missing: $1" >&2; return 1; }; }
if ! need yt-dlp; then
    pip install -q yt-dlp
fi
if ! need ffmpeg; then
    if command -v apt-get >/dev/null; then
        apt-get update -qq
        apt-get install -y -q ffmpeg
    else
        echo "ffmpeg not found and apt-get unavailable" >&2
        exit 1
    fi
fi

mkdir -p "$OUT"
cd "$OUT"

# 1. Download mp4 + auto-subs as SRT. -f 'best[height<=480]' picks the
#    smallest progressive stream that still has both audio and video, so
#    we don't need ffmpeg to merge separate streams.
echo "[1/3] Downloading video + subs into $OUT ..."
yt-dlp \
    -f 'best[height<=480]' \
    -o 'video.%(ext)s' \
    --write-auto-subs \
    --sub-lang en \
    --convert-subs srt \
    "$URL"

# 2. Dedupe rolling auto-captions into a clean transcript.
echo "[2/3] Deduping captions ..."
python "$(dirname "$0")/dedupe_srt.py" video.en.srt > transcript.txt
echo "    transcript.txt: $(wc -c < transcript.txt) chars"

# 3. Keyframes — 1 frame every 30 s, scaled to 720 wide. These let you
#    inspect chart screenshots from the video when the spoken transcript
#    is ambiguous (e.g., the trader points at "this rejection candle
#    here" — pull the frame at that timestamp from the SRT).
echo "[3/3] Extracting keyframes (1 / 30s) ..."
mkdir -p frames
ffmpeg -y -loglevel error \
    -i video.mp4 \
    -vf 'fps=1/30,scale=720:-1' \
    'frames/frame_%04d.png'
echo "    extracted $(ls frames/ | wc -l) frames"

echo
echo "Done. Files:"
ls -la "$OUT" | grep -v '^d' | awk '{print "  " $0}'
