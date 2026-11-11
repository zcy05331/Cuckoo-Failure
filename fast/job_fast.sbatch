#!/bin/bash
#SBATCH --job-name=cuckoo-fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%A_%a.log
# Array task b runs prime chunks BASE+4b .. BASE+4b+3 (up to LAST), one process per core.
# Each chunk appends to its own file and skips primes already there, so a resubmission
# (or another node) simply continues. Export D, QUERIES, CHUNK, TAG, BASE, LAST.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p out
if grep -q avx512ifma /proc/cpuinfo; then BIN=./cuckoo_fast_ifma; else BIN=./cuckoo_fast_cpu; fi
echo "$(hostname) $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) -> $BIN"
for k in 0 1 2 3; do
  c=$((BASE + 4*SLURM_ARRAY_TASK_ID + k))
  [ "$c" -gt "$LAST" ] && continue
  $BIN "$D" "$QUERIES" $((c*CHUNK)) "$CHUNK" --out "out/${TAG}_${c}.txt" 2> "logs/${TAG}_${c}.err" &
done
wait
