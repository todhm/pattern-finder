# YouTube → Strategy: video extraction workflow

How we pull a trading-strategy YouTube video into the repo well enough
to implement the rules in code. The Trade Sharp ORB page
(`pages/18_TradeSharp_ORB.py`) was built using exactly this pipeline.

The web tools available to Claude/this codebase **cannot read YouTube
transcripts directly** — `WebFetch` only sees the page chrome, not the
caption stream. So we download the artifact locally, dedupe the rolling
auto-captions, and (when the spoken word is ambiguous) sample frames
to inspect chart screenshots.

## What you get

Running `scripts/fetch_video_strategy.sh <url> <out_dir>` produces:

| File | Purpose |
|---|---|
| `video.mp4` | The actual video, ~20–30 MB at 480p. Watch directly when needed. |
| `video.en.srt` | Raw auto-generated English captions. |
| `transcript.txt` | One-paragraph deduped transcript (~30–60 KB) suitable for reading or pasting into a chat. |
| `frames/frame_NNNN.png` | Keyframes sampled every 30 s, scaled to 720 wide. Useful for inspecting chart screenshots when the trader says "the rejection candle *here*". |

## Why dedupe?

YouTube's auto-captions emit **rolling overlapping** lines:

```
00:00:01 → 00:00:04   Most traders find ORB and think it's the
00:00:04 → 00:00:07   Most traders find ORB and think it's the holy grail strategy.
00:00:07 → 00:00:10   holy grail strategy. But in 2025 and
```

Naive `.text` concatenation gives you the same sentence 2–3 times. The
`dedupe_srt.py` helper drops consecutive duplicates and substring
overlaps:

- If the next line == the previous line → skip
- If the next line is wholly contained in the previous → skip
- If the previous line is wholly contained in the next → keep the longer

After dedupe, the transcript reads cleanly start-to-finish.

## Running it

The script is built to run **inside the backtester container** so
`yt-dlp` and `ffmpeg` install without touching the host:

```bash
docker compose exec backtester bash scripts/fetch_video_strategy.sh \
    "https://www.youtube.com/watch?v=W5Hxv3hL3vY" /tmp/orb_video
```

Output ends up at `/tmp/orb_video/` inside the container. To copy the
transcript out:

```bash
docker compose cp \
    backtester:/tmp/orb_video/transcript.txt \
    ./docs/research/transcripts/W5Hxv3hL3vY.txt
```

If the script complains it can't `pip install yt-dlp` or `apt-get
install ffmpeg`, your container needs network access — these are one-
time installs.

## The two helpers

```
scripts/
├── fetch_video_strategy.sh   # 3-step pipeline: download → dedupe → frames
└── dedupe_srt.py             # SRT → clean text, used by the shell script
```

`dedupe_srt.py` is also runnable standalone if you already have an SRT:

```bash
python scripts/dedupe_srt.py video.en.srt > transcript.txt
```

## When to use frames

Reading the transcript covers ~80% of strategy research. The remaining
20% — where the trader points at the screen and says "see *this* candle
right *here*" — needs the visual. After running the pipeline you have
1 frame per 30 seconds in `frames/`. To find the frame at a specific
timestamp, pull the corresponding entry from the SRT and divide by 30:

```bash
# Find what's said around 5:30 in the video
grep -A2 '00:05:[23]' video.en.srt
# Look at the matching frame
open frames/frame_0011.png    # 5:30 / 30s = frame 11
```

## Practical limits

- **Auto-captions can be wrong.** Words like "OHLC" sometimes come
  out as "O HLC" or "OH LC". Numeric prices ("1.5R") sometimes drop
  to "1 5 R". Re-watch the relevant 10 seconds when the spoken rule
  is critical.
- **Unlisted / private / age-gated videos** require yt-dlp cookies.
  Pass `--cookies-from-browser <chrome|firefox>` if needed.
- **Long videos (>1h)** produce a transcript that's too big for a
  single context window — split by section before pasting into chat.
- **Rate limits** — YouTube throttles after ~5–10 video downloads in
  a short window. Space out requests if doing batch research.

## How this informed the Trade Sharp ORB page

Specific rules I extracted from the W5Hxv3hL3vY transcript that
became code in `pattern/adapters/tradesharp_orb.py`:

| Spoken rule | Code |
|---|---|
| "15-minute ORB" | `box_minutes=15` (= 3 × 5m bars) |
| "5-minute time frame, more accurate than 15" | strategy runs on 5m bars |
| "Most popular is waiting for 15-minute close" / "first breakout = slapped in the face" | Phase 1 records the breakout but does **not** enter |
| "Liquidity grab back into the open range" | Phase 2 requires `lows[i] <= box_high` |
| "Rejection candle / support candle" | Phase 3 requires `closes[i] > opens[i]` after pullback low |
| "Wait for next candle to break the high" | Phase 4 fires on `highs[i] > rejection_high` |
| "Stop loss below" the rejection / "back inside of the open range" | `stop_loss = rejection_low` |
| "1:1 to 1:2 R/R" | `target_r_multiple` default 1.5 |
| "Higher time frame bias … previous daily candle did" | `require_daily_bullish=True`, gates on `prev_close > prev_open` |
| "Doesn't work too well in London open … NYSE open is the best" | docstring restricts to NY session by default |

Each row is auditable against the saved transcript — that's the value
of having `transcript.txt` checked in alongside the implementation.
