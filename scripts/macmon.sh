#!/usr/bin/env bash
# Mac health monitor for long ML runs on a fanless MacBook Air: every 15 s logs CPU load, thermal state, memory, swap,
# battery temperature and power source to runs/macmon.log. Sets runs/.hot when the Mac is too busy or too warm
# (CPU > 80 % in one sample, or thermal state >= 2 "serious"); batch jobs (vision.evaluate) wait while it exists.
# Thermal state is macOS' own ProcessInfo.thermalState (0 nominal, 1 fair, 2 serious, 3 critical) — no sudo needed.
# Usage: scripts/macmon.sh            (Ctrl-C to stop; run in a spare terminal tab or in the background)
#        scripts/macmon.sh --once     (one line, for a quick check)
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/runs/.bin"
probe="$root/runs/.bin/thermal"
if [ ! -x "$probe" ]; then
  printf 'import Foundation\nprint(ProcessInfo.processInfo.thermalState.rawValue)\n' > "$root/runs/.bin/thermal.swift"
  swiftc -O "$root/runs/.bin/thermal.swift" -o "$probe" 2>/dev/null || { echo "✖ swiftc не найден — xcode-select --install"; exit 1; }
fi
log="$root/runs/macmon.log"; hot="$root/runs/.hot"
CPU_LIMIT="${CPU_LIMIT:-80}"
busy=0; calm=0

sample() {
  local idle cpu therm mem swap batt power
  idle=$(top -l 2 -s 1 -n 0 2>/dev/null | grep "CPU usage" | tail -1 | sed -E 's/.* ([0-9.]+)% idle.*/\1/')
  cpu=$(printf '%.0f' "$(echo "100 - ${idle:-100}" | bc)")
  therm=$("$probe" 2>/dev/null || echo "?")
  mem=$(memory_pressure 2>/dev/null | sed -n 's/System-wide memory free percentage: //p')
  swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+M).*/\1/')
  batt=$(ioreg -rn AppleSmartBattery 2>/dev/null | sed -n 's/.*"Temperature" = \([0-9]*\).*/\1/p' | head -1)
  batt=$(printf '%.1f' "$(echo "${batt:-0} / 100" | bc -l)")
  power=$(pmset -g batt | head -1 | sed -E "s/.*'(.*)'.*/\1/" | sed 's/ Power//')
  local pct; pct=$(pmset -g batt | grep -o '[0-9]*%' | head -1)
  echo "$(date +%H:%M:%S) cpu=${cpu}% thermal=${therm} mem_free=${mem} swap_used=${swap} batt=${batt}C power=${power} ${pct}"
  CPU=$cpu; THERM=$therm; POWER=$power
}

if [ "${1:-}" = "--once" ]; then sample; exit 0; fi
# Only remove our own marker; a killed monitor is detected by its stale heartbeat.
cleanup() { [ "$(cat "$hot" 2>/dev/null)" != "$$" ] || rm -f "$hot"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "macmon: журнал $log, флаг перегрузки $hot (CPU > ${CPU_LIMIT}% или thermal ≥ 2). Ctrl-C — стоп."
while :; do
  line=$(sample); echo "$line" | tee -a "$log"
  CPU=$(echo "$line" | sed -E 's/.*cpu=([0-9]+)%.*/\1/'); THERM=$(echo "$line" | sed -E 's/.*thermal=([0-9?]).*/\1/')
  if [ "$THERM" != "?" ] && [ "$THERM" -ge 2 ]; then busy=1; fi
  if [ "${CPU:-0}" -gt "$CPU_LIMIT" ]; then busy=$((busy + 1)); calm=0; else calm=$((calm + 1)); [ "$busy" -lt 1 ] && busy=0; fi
  if [ "$busy" -ge 1 ] && [ ! -f "$hot" ]; then
    echo "$$" > "$hot"; echo "⚠ $(date +%H:%M) Мак перегружен/греется — пакетные прогоны на паузе"
    osascript -e 'display notification "CPU > 80% или нагрев — прогоны на паузе" with title "Datathon"' >/dev/null 2>&1 || true
  fi
  # Refresh even while a hot interval continues; stale flags must never look live.
  [ ! -f "$hot" ] || { echo "$$" > "$hot"; }
  if [ -f "$hot" ] && [ "$calm" -ge 2 ] && { [ "$THERM" = "?" ] || [ "$THERM" -le 1 ]; }; then
    rm -f "$hot"; busy=0; echo "✔ $(date +%H:%M) остыл — прогоны продолжаются"
  fi
  case "$line" in *"power=Battery"*) echo "  ⚠ работа от батареи — подключи зарядку";; esac
  sleep 6
done
