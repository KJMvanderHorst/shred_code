#!/usr/bin/env bash
# Time each setup on a fixed workload of 6 identical jobs (each: 3 runs of configs/experiment/speed_benchmark.yaml).
# Appends setup,rep,makespan_s to outputs/speed_benchmark/timings.csv; run.py output goes to bench.log.
#   experiments/speed_benchmark/bench.sh                 # all setups, 2 repeats, round-robin
#   REPS=1 SETUPS="cpu_3x2" experiments/speed_benchmark/bench.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=outputs/speed_benchmark; mkdir -p "$OUT"
CSV=$OUT/timings.csv; LOG=$OUT/bench.log
[ -f "$CSV" ] || echo "setup,rep,makespan_s" > "$CSV"
REPS=${REPS:-2}
SETUPS=${SETUPS:-"mps_1 mps_2 mps_3 cpu_1x6 cpu_1x1 cpu_2x3 cpu_3x2 cpu_6x1 hybrid"}
uv sync -q  # so uv start-up does not count against the first setup

# jobs <device> <parallel N> <threads T> <setup> <rep> <job ids...>
jobs() {
  local dev=$1 n=$2 t=$3 setup=$4 rep=$5; shift 5
  printf '%s\n' "$@" | OMP_NUM_THREADS=$t xargs -P "$n" -I{} \
    uv run python scripts/run.py +experiment=speed_benchmark device="$dev" \
      name="speed_benchmark/$setup/r$rep/job{}" >> "$LOG" 2>&1
}

for rep in $(seq 1 "$REPS"); do
  for setup in $SETUPS; do
    rm -rf "$OUT/$setup/r$rep"  # fresh outputs, or the resumable runner skips the work
    start=$(python3 -c 'import time; print(time.time())')
    case $setup in
      mps_*)   jobs mps "${setup#mps_}" 8 "$setup" "$rep" 1 2 3 4 5 6 ;;  # T irrelevant-ish on MPS; torch default
      cpu_*)   nt=${setup#cpu_}; jobs cpu "${nt%x*}" "${nt#*x}" "$setup" "$rep" 1 2 3 4 5 6 ;;
      hybrid)  jobs mps 2 8 "$setup" "$rep" 1 2 & jobs cpu 4 1 "$setup" "$rep" 3 4 5 6; wait ;;
    esac
    end=$(python3 -c 'import time; print(time.time())')
    python3 -c "print('$setup,$rep,%.1f' % ($end - $start))" >> "$CSV"
  done
done
