#!/usr/bin/env bash
set -euo pipefail

DIMENSIONS=("1D" "2D")
MODES=("SEQ" "PAR")
TOTAL_THREADS_LIST=("1" "2" "4" "8") #modify in cluster

# Per-thread baseline problem size (matches your generator idea)
BASE_N=1000
BASE_NNZ=100000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
RUN_DIR="$PROJECT_ROOT/build"
RANDOM_DIR="$PROJECT_ROOT/random_matrices"

RESULTS_DIR="$SCRIPT_DIR/results_weak_hybrid/"
CSV_DIR="$RESULTS_DIR/summary_csvs/"
mkdir -p "$RESULTS_DIR" "$CSV_DIR"

echo "Results will be stored in: $RESULTS_DIR"
echo "CSV summaries will be stored in: $CSV_DIR"
echo

MPI_LAUNCH=(mpirun)
if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
  MPI_LAUNCH+=( -hostfile "$PBS_NODEFILE" )
fi

pushd "$RUN_DIR" >/dev/null

for DIMENSION in "${DIMENSIONS[@]}"; do
  for MODE in "${MODES[@]}"; do

    OUT_FILE="$RESULTS_DIR/all_configs__hybrid__${DIMENSION}__${MODE}.log"
    : > "$OUT_FILE"

    SUMMARY_CSV="$CSV_DIR/summary__weak_hybrid__${DIMENSION}__${MODE}.csv"
    echo "matrix,n,nnz,total_threads,np,omp,dimension,mode,avg_time_s,p90_time_s,baseline_s,weak_speedup,weak_efficiency" >"$SUMMARY_CSV"

    BASELINE_S=""

    for TOTAL in "${TOTAL_THREADS_LIST[@]}"; do
      if (( TOTAL <= 16 )); then
        NP="$TOTAL"
        OMP=1
      else
        NP=16
        if (( TOTAL % NP != 0 )); then
          echo "WARN: TOTAL_THREADS=$TOTAL not divisible by NP=$NP, skipping" | tee -a "$OUT_FILE"
          continue
        fi
        OMP=$(( TOTAL / NP ))
      fi

      N=$(( BASE_N * TOTAL ))
      NNZ=$(( BASE_NNZ * TOTAL ))

      MATRIX_FILE="random_${N}_nnz${NNZ}.mtx"
      MATRIX_PATH="$RANDOM_DIR/$MATRIX_FILE"

      if [[ ! -f "$MATRIX_PATH" ]]; then
        echo "WARN: Missing matrix file: $MATRIX_PATH (generate it for TOTAL_THREADS=$TOTAL)" | tee -a "$OUT_FILE"
        continue
      fi

      echo "============================================================" >>"$OUT_FILE"
      echo "WEAK HYBRID RUN: TOTAL_THREADS=$TOTAL NP=$NP OMP=$OMP  N=$N NNZ=$NNZ  DIMENSION=$DIMENSION MODE=$MODE" >>"$OUT_FILE"
      echo "MATRIX: $MATRIX_FILE" >>"$OUT_FILE"
      echo "============================================================" >>"$OUT_FILE"

      if [[ "$MODE" == "PAR" ]]; then
        export OMP_NUM_THREADS="$OMP"
      else
        unset OMP_NUM_THREADS || true
      fi

      if [[ "$MODE" == "PAR" ]]; then
        echo "OMP_NUM_THREADS=$OMP_NUM_THREADS" >>"$OUT_FILE"
      fi

      echo "---- MATRIX: $MATRIX_FILE ----" | tee -a "$OUT_FILE"

      RUN_OUTPUT="$(
        "${MPI_LAUNCH[@]}" -np "$NP" \
          ./spmv "../random_matrices/$MATRIX_FILE" "$DIMENSION" "$MODE" \
          2>&1 | tee -a "$OUT_FILE"
      )"

      echo | tee -a "$OUT_FILE"

      AVG_TIME_S="$(awk '/SpMV Time over/{flag=1;next} flag && $1=="Avg:"{gsub("s","",$2); print $2; exit}' <<<"$RUN_OUTPUT")"
      P90_TIME_S="$(awk '/SpMV Time over/{flag=1;next} flag && $1=="P90:"{gsub("s","",$2); print $2; exit}' <<<"$RUN_OUTPUT")"

      if [[ -z "${AVG_TIME_S:-}" ]]; then
        echo "WARN: Could not parse Avg time for $MATRIX_FILE (TOTAL=$TOTAL NP=$NP OMP=$OMP $DIMENSION $MODE)" | tee -a "$OUT_FILE"
        continue
      fi
      if [[ -z "${P90_TIME_S:-}" ]]; then
        P90_TIME_S=""
      fi

      if (( TOTAL == 1 )); then
        BASELINE_S="$AVG_TIME_S"
      fi

      if [[ -z "${BASELINE_S:-}" ]]; then
        echo "WARN: Baseline not set yet; ensure TOTAL_THREADS=1 runs first." | tee -a "$OUT_FILE"
        continue
      fi

      WEAK_SPEEDUP="$(awk -v b="$BASELINE_S" -v t="$AVG_TIME_S" 'BEGIN{printf "%.6f", b/t}')"
      WEAK_EFF="$WEAK_SPEEDUP"

      echo "${MATRIX_FILE},${N},${NNZ},${TOTAL},${NP},${OMP},${DIMENSION},${MODE},${AVG_TIME_S},${P90_TIME_S},${BASELINE_S},${WEAK_SPEEDUP},${WEAK_EFF}" >>"$SUMMARY_CSV"

    done
  done
done

popd >/dev/null

echo
echo "All hybrid weak-scaling runs completed."
echo "Logs: $RESULTS_DIR"
echo "Summaries: $CSV_DIR"
