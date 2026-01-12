#!/usr/bin/env bash
set -euo pipefail

MATRICES=(
  "494_bus.mtx"
  "1138_bus.mtx"
  "bcspwr06.mtx"
  "bcspwr08.mtx"
  "bcspwr10.mtx"
)

DIMENSIONS=("1D" "2D")
MODES=("SEQ" "PAR")
PROCS_LIST_DEFAULT=("1" "2" "4" "8") #modify in cluster

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
RUN_DIR="$PROJECT_ROOT/build"

RESULTS_DIR="$SCRIPT_DIR/results_strong/"
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/summary_csvs"

echo "Results will be stored in: $RESULTS_DIR"
echo

echo "Building project (cd .. && ./scripts/build_mac.sh) ..."
pushd "$PROJECT_ROOT" >/dev/null
./scripts/build_mac.sh #MODIFY IN CLUSTER
popd >/dev/null
echo "Build done."
echo

pushd "$RUN_DIR" >/dev/null

# One file per (DIMENSION, MODE, NP) — all matrices appended inside
for DIMENSION in "${DIMENSIONS[@]}"; do
  for MODE in "${MODES[@]}"; do

    #one CSV per (DIMENSION, MODE)
    SUMMARY_CSV="$RESULTS_DIR/summary_csvs/summary__${DIMENSION}__${MODE}.csv"
    if [[ ! -f "$SUMMARY_CSV" ]]; then
      echo "matrix,dimension,mode,np,avg_time_s,p90_time_s,baseline_s,speedup,efficiency" >"$SUMMARY_CSV"
    fi

    for NP in "${PROCS_LIST_DEFAULT[@]}"; do

      if (( NP < 1 || NP > 128 )); then
        echo "Skipping NP=$NP (out of range 1..128)"
        continue
      fi

      if [[ "$MODE" == "PAR" ]]; then
        export OMP_NUM_THREADS=4
      else
        unset OMP_NUM_THREADS || true
      fi

      for MATRIX in "${MATRICES[@]}"; do

        OUT_FILE="$RESULTS_DIR/${MATRIX%.mtx}.log"

        echo "============================================================" >>"$OUT_FILE"
        echo "RUN: MATRIX=$MATRIX DIMENSION=$DIMENSION MODE=$MODE NP=$NP" >>"$OUT_FILE"
        if [[ "$MODE" == "PAR" ]]; then
          echo "OMP_NUM_THREADS=$OMP_NUM_THREADS" >>"$OUT_FILE"
        fi
        echo "============================================================" >>"$OUT_FILE"

        echo "---- MATRIX: $MATRIX ----" | tee -a "$OUT_FILE"

        RUN_OUTPUT="$(mpirun -np "$NP" ./spmv "../../matrix_mkt/$MATRIX" "$DIMENSION" "$MODE" 2>&1 | tee -a "$OUT_FILE")"
        echo | tee -a "$OUT_FILE"

        #to compute efficiency vs speedup
        # Parse the "Avg: X s" under "SpMV Time over 100 iterations:"
        AVG_TIME_S="$(awk '/SpMV Time over 100 iterations:/{flag=1;next} flag && $1=="Avg:"{gsub("s","",$2); print $2; exit}' <<<"$RUN_OUTPUT")"

        if [[ -z "${AVG_TIME_S:-}" ]]; then
          echo "WARN: Could not parse Avg time for $MATRIX NP=$NP $DIMENSION $MODE" | tee -a "$OUT_FILE"
          continue
        fi

        # Store baseline (NP=1) per (matrix, dimension, mode) in a variable name
        BASELINE_FILE="$RESULTS_DIR/baselines__${DIMENSION}__${MODE}.csv"
        touch "$BASELINE_FILE"

        # Save / load baseline (NP=1) per matrix
        if (( NP == 1 )); then
          grep -v "^${MATRIX}," "$BASELINE_FILE" > "${BASELINE_FILE}.tmp" || true
          mv "${BASELINE_FILE}.tmp" "$BASELINE_FILE"
          echo "${MATRIX},${AVG_TIME_S}" >> "$BASELINE_FILE"
          BASELINE_S="$AVG_TIME_S"
        else
          BASELINE_S="$(awk -F',' -v m="$MATRIX" '$1==m {print $2; exit}' "$BASELINE_FILE")"
        fi


        SPEEDUP="$(awk -v b="$BASELINE_S" -v t="$AVG_TIME_S" 'BEGIN{printf "%.6f", b/t}')"
        EFFICIENCY="$(awk -v s="$SPEEDUP" -v p="$NP" 'BEGIN{printf "%.6f", s/p}')"
        P90_TIME_S="$(awk '/SpMV Time over/{flag=1;next} flag && $1=="P90:"{gsub("s","",$2); print $2; exit}' <<<"$RUN_OUTPUT")"

        echo "${MATRIX},${DIMENSION},${MODE},${NP},${AVG_TIME_S},${P90_TIME_S},${BASELINE_S},${SPEEDUP},${EFFICIENCY}" >>"$SUMMARY_CSV"
      done

    done
  done
done

popd >/dev/null

rm -f "$RESULTS_DIR"/baselines__*.csv

echo
echo "All runs completed."
echo "Logs: $RESULTS_DIR"
