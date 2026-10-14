#!/usr/bin/env bash
# DATATHON'26 timeline, Astana time. Sourced by clock.sh, autopush.sh and msg.sh — change times only here.
# Rehearsal: put overrides in scripts/timeline.local.sh (git-ignored), e.g. DT_START="2026-10-15 20:00".
DT_START="2026-10-17 12:00"    # 24-hour clock starts (dataset revealed at 11:00)
DT_MID="2026-10-17 22:00"      # midpoint: direction locked, pitch skeleton exists
DT_FREEZE="2026-10-18 08:00"   # model freeze: final model and submission; after it only fixes, deck, README
DT_FINAL="2026-10-18 11:30"    # final push
DT_END="2026-10-18 12:00"      # laptops down, submissions close
DT_PITCH="2026-10-18 13:00"    # pitching to the jury (13:00-17:00)

_dt_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[ -f "$_dt_dir/timeline.local.sh" ] && . "$_dt_dir/timeline.local.sh"

# epoch seconds of "YYYY-MM-DD HH:MM" (BSD date on macOS, GNU date on Linux and Git Bash)
dt_epoch() { date -j -f '%Y-%m-%d %H:%M:%S' "$1:00" +%s 2>/dev/null || date -d "$1" +%s; }
# "3ч05м" / "42м" until the epoch $1
dt_left() {
  local m=$(( ($1 - $(date +%s)) / 60 ))
  if [ "$m" -ge 60 ]; then printf '%dч%02dм' $((m / 60)) $((m % 60)); else echo "${m}м"; fi
}
dt_past() { [ "$(date +%s)" -ge "$(dt_epoch "$1")" ]; }
