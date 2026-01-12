#!/usr/bin/env bash
set -euo pipefail

# Weak-scaling configs: problem size grows with NP
CONFIG_N=(1000 2000 4000 8000) #modify in cluster till 128
CONFIG_NNZ=(100000 200000 400000 800000)
CONFIG_NP=(1 2 4 8)

DIMENSIONS=("1D" "2D")
MODES=("SEQ" "PAR")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
RUN_DIR="$PROJECT_ROOT/build"                 # ./spmv is here (deliverable2/build)
RANDOM_DIR="$PROJECT_ROOT/random_matrices"       # deliverable2/random_matrices

RESULTS_DIR="$SCRIPT_DIR/results_weak/"
mkdir -p "$RESULTS_DIR"

CSV_DIR="$RESULTS_DIR/summary_csvs/"
mkdir -p "$CSV_DIR"

echo "Results will be stored in: $RESULTS_DIR"
echo "CSV summaries will be stored in: $CSV_DIR"
echo

echo "Building project (cd .. && ./scripts/build_mac.sh) ..." #modify in cluster
pushd "$PROJECT_ROOT" >/dev/null
./scripts/build_mac.sh
popd >/dev/null
echo "Build done."
echo

pushd "$RUN_DIR" >/dev/null


for DIMENSION in "${DIMENSIONS[@]}"; do
  for MODE in "${MODES[@]}"; do

    SUMMARY_CSV="$CSV_DIR/summary__weak__${DIMENSION}__${MODE}.csv"
    echo "matrix,n,nnz,np,dimension,mode,avg_time_s,p90_time_s,baseline_s,weak_speedup,weak_efficiency" >"$SUMMARY_CSV"

    BASELINE_S=""

    for i in "${!CONFIG_N[@]}"; do
      N="${CONFIG_N[$i]}"
      NNZ="${CONFIG_NNZ[$i]}"
      NP="${CONFIG_NP[$i]}"

      if (( NP < 1 || NP > 128 )); then
        echo "Skipping NP=$NP (out of range 1..128)"
        continue
      fi

      MATRIX_FILE="random_${N}_nnz${NNZ}.mtx"
      MATRIX_PATH="$RANDOM_DIR/$MATRIX_FILE"

      if [[ ! -f "$MATRIX_PATH" ]]; then
        echo "WARN: Missing matrix file: $MATRIX_PATH"
        continue
      fi

      OUT_FILE="$RESULTS_DIR/all_configs__${DIMENSION}__${MODE}.log"

      echo "============================================================" >>"$OUT_FILE"
      echo "WEAK RUN: N=$N NNZ=$NNZ NP=$NP DIMENSION=$DIMENSION MODE=$MODE" >>"$OUT_FILE"
      echo "MATRIX: $MATRIX_FILE" >>"$OUT_FILE"
      echo "============================================================" >>"$OUT_FILE"

      if [[ "$MODE" == "PAR" ]]; then
        export OMP_NUM_THREADS=4
        echo "OMP_NUM_THREADS=$OMP_NUM_THREADS" >>"$OUT_FILE"
      else
        unset OMP_NUM_THREADS || true
      fi

      echo "---- MATRIX: $MATRIX_FILE ----" | tee -a "$OUT_FILE"

      RUN_OUTPUT="$(mpirun -np "$NP" ./spmv "../random_matrices/$MATRIX_FILE" "$DIMENSION" "$MODE" 2>&1 | tee -a "$OUT_FILE")"

      echo | tee -a "$OUT_FILE"

      AVG_TIME_S="$(awk '/SpMV Time over 100 iterations:/{flag=1;next} flag && $1=="Avg:"{gsub("s","",$2); print $2; exit}' <<<"$RUN_OUTPUT")"

      if [[ -z "${AVG_TIME_S:-}" ]]; then
        echo "WARN: Could not parse Avg time for $MATRIX_FILE (N=$N NNZ=$NNZ NP=$NP $DIMENSION $MODE)" | tee -a "$OUT_FILE"
        continue
      fi

      # Baseline for weak scaling: the NP=1 configuration (first / smallest problem)
      if (( NP == 1 )); then
        BASELINE_S="$AVG_TIME_S"
      fi

      if [[ -z "${BASELINE_S:-}" ]]; then
        echo "WARN: Baseline not set yet; ensure NP=1 config exists and runs first." | tee -a "$OUT_FILE"
        continue
      fi

      # Weak scaling: ideal is constant time, so efficiency = baseline / current
      WEAK_SPEEDUP="$(awk -v b="$BASELINE_S" -v t="$AVG_TIME_S" 'BEGIN{printf "%.6f", b/t}')"
      WEAK_EFF="$(awk -v ws="$WEAK_SPEEDUP" 'BEGIN{printf "%.6f", ws}')"
      P90_TIME_S="$(awk '/SpMV Time over/{flag=1;next} flag && $1=="P90:"{gsub("s","",$2); print $2; exit}' <<<"$RUN_OUTPUT")"

      echo "${MATRIX_FILE},${N},${NNZ},${NP},${DIMENSION},${MODE},${AVG_TIME_S},${P90_TIME_S},${BASELINE_S},${WEAK_SPEEDUP},${WEAK_EFF}" >>"$SUMMARY_CSV"

    done
  done
done

popd >/dev/null

echo
echo "All weak-scaling runs completed."
echo "Logs: $RESULTS_DIR"
echo "Summaries: $CSV_DIR"
