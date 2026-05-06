#!/usr/bin/env bash
# 유튜브 튜토리얼 영상 + PDF를 마크다운 정리용 자료로 변환하는 워크플로 스크립트
#
# 사용 예:
#   ./extract_youtube_tutorial.sh \
#       "https://www.youtube.com/watch?v=m5zu_X-_51I" \
#       "/Users/apple/development/pattern-finder/docs/strategy_notes" \
#       "/Users/apple/development/pattern-finder/20250611 Candlestick Pattern Reference Chart.pdf" \
#       "/Users/apple/development/pattern-finder/ChartPatternsv2.pdf"
#
# 종속성 (없으면 자동 설치 시도):
#   - poppler  (pdftoppm, pdftotext, pdfinfo)  : brew install poppler
#   - yt-dlp                                    : brew install yt-dlp
#   - ffmpeg                                    : brew install ffmpeg
#
# 출력:
#   <out_dir>/
#     ├── images/video_frames/{HHMMSS}_{label}.jpg
#     ├── images/pdf_<name1>/page-NNN.jpg
#     ├── images/pdf_<name2>/page-NNN.jpg
#     ├── transcript.<lang>.vtt
#     ├── pdf_<name>.txt        (텍스트 추출본)
#     └── video_lowres.mp4

set -euo pipefail

YT_URL="${1:?usage: $0 <youtube_url> <out_dir> [pdf...]}"
OUT_DIR="${2:?usage: $0 <youtube_url> <out_dir> [pdf...]}"
shift 2

mkdir -p "$OUT_DIR/images/video_frames"

# ────────────────────────────────────────────────────────────────────
# 0) 종속성 설치 (멱등)
# ────────────────────────────────────────────────────────────────────
ensure_brew_pkg() {
  local pkg=$1; local bin=$2
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "[install] brew install $pkg"
    brew install "$pkg"
  fi
}
ensure_brew_pkg poppler   pdftoppm
ensure_brew_pkg yt-dlp    yt-dlp
ensure_brew_pkg ffmpeg    ffmpeg

# pdftoppm이 /usr/local/bin에만 깔려있고 Read 도구가 /opt/homebrew/bin을 본다면 심볼릭 링크로 보정
if [[ -x /usr/local/bin/pdftoppm && ! -e /opt/homebrew/bin/pdftoppm ]]; then
  ln -sf /usr/local/bin/pdftoppm /opt/homebrew/bin/pdftoppm 2>/dev/null || true
fi

# ────────────────────────────────────────────────────────────────────
# 1) 자막 다운로드 (VTT, 영문 자동/공식)
# ────────────────────────────────────────────────────────────────────
echo "[1/4] downloading subtitles..."
yt-dlp \
  --skip-download \
  --write-auto-sub --write-sub \
  --sub-lang en --sub-format vtt \
  -o "$OUT_DIR/transcript.%(ext)s" \
  "$YT_URL"

# ────────────────────────────────────────────────────────────────────
# 2) 저해상 비디오 다운로드 (프레임 추출용 — 풀해상은 불필요)
# ────────────────────────────────────────────────────────────────────
VIDEO_PATH="$OUT_DIR/video_lowres.mp4"
if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "[2/4] downloading low-res video..."
  yt-dlp \
    -f "worst[height<=360]/worst" \
    -o "$VIDEO_PATH" \
    "$YT_URL"
else
  echo "[2/4] video already downloaded, skipping."
fi

# ────────────────────────────────────────────────────────────────────
# 3) PDF → 텍스트 + 페이지 이미지
# ────────────────────────────────────────────────────────────────────
echo "[3/4] processing PDFs..."
for pdf in "$@"; do
  [[ -f "$pdf" ]] || { echo "  skip (not found): $pdf"; continue; }
  base=$(basename "$pdf" .pdf | tr ' ' '_')
  img_dir="$OUT_DIR/images/pdf_${base}"
  mkdir -p "$img_dir"
  echo "  - $base → text"
  pdftotext "$pdf" "$OUT_DIR/pdf_${base}.txt" || true
  echo "  - $base → images (1280px equiv, JPEG)"
  pdftoppm -r 150 -jpeg "$pdf" "$img_dir/page" || true
done

# ────────────────────────────────────────────────────────────────────
# 4) 비디오 프레임 추출 헬퍼
#
#   사용법:
#     extract_frame "00:05:30" "51day_metrics"
#
#   결과:
#     <OUT_DIR>/images/video_frames/000530_51day_metrics.jpg  (1280px wide)
# ────────────────────────────────────────────────────────────────────
extract_frame() {
  local ts="$1"  ; local label="$2"
  local fname="${ts//:/}_${label}.jpg"
  ffmpeg -y -ss "$ts" -i "$VIDEO_PATH" \
         -frames:v 1 -q:v 2 \
         -vf "scale=1280:-1" \
         "$OUT_DIR/images/video_frames/$fname" 2>/dev/null
  echo "  ✓ $fname"
}
export -f extract_frame
export VIDEO_PATH OUT_DIR

# ────────────────────────────────────────────────────────────────────
# 5) (선택) Ross Cameron 영상 기준 핵심 25+ 타임스탬프 일괄 추출
#    다른 영상에서 재사용할 때는 이 배열을 영상에 맞게 교체.
# ────────────────────────────────────────────────────────────────────
TIMESTAMPS=(
  "00:00:30:intro"
  "00:05:30:51day_equity_curve"
  "00:07:30:pl_ratio_chart"
  "00:08:30:risk_reward_table"
  "00:21:30:three_core_components"
  "00:23:30:positive_feedback_loop"
  "00:27:50:price_range_example"
  "00:29:00:price_range_chart"
  "00:30:50:five_criteria_full"
  "00:31:30:scanner_lowfloat"
  "00:33:30:bull_flag_intro"
  "00:34:50:bull_flag_pattern_complete"
  "00:36:00:bull_flag_pullback"
  "00:36:50:bull_flag_with_entry_arrow"
  "00:37:20:bull_flag_full_setup"
  "00:38:40:bullflag_HoD_label"
  "00:39:50:bull_flag_real1_zoomed"
  "00:40:30:bull_flag_real2"
  "00:42:00:dragonfly_doji"
  "00:43:30:bullflag_example2"
  "00:50:00:atnf_98k_day"
  "00:51:30:position_management"
  "00:56:00:revenge_trading_loop"
  "01:03:30:quarter_cushion_strategy"
  "01:07:00:final_pl_curves"
  "01:11:30:adding_to_winners"
  "01:13:30:double_position"
  "01:32:00:metrics_review"
  "01:35:30:scaling_plan"
)

echo "[4/4] extracting key frames..."
for entry in "${TIMESTAMPS[@]}"; do
  ts="${entry%:*}"
  label="${entry##*:}"
  extract_frame "$ts" "$label"
done

echo
echo "✅ Done. Output: $OUT_DIR"
echo "   - transcript.en.vtt"
echo "   - video_lowres.mp4"
echo "   - images/video_frames/  ($(ls "$OUT_DIR/images/video_frames" | wc -l | tr -d ' ') frames)"
echo "   - images/pdf_*/         (PDF pages)"
