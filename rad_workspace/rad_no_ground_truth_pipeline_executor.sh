#!/usr/bin/env bash
set -euo pipefail
# --- auto-detect repo root and cd to rad_workspace ---
START_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT="$(git -C "$START_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  d="$START_DIR"
  while [[ "$d" != "/" ]]; do
    [[ -d "$d/.git" ]] && { REPO_ROOT="$d"; break; }
    d="$(dirname "$d")"
  done
fi

[[ -n "$REPO_ROOT" ]] || { echo "ERROR: could not detect repo root" >&2; exit 1; }

export REPO_ROOT
RAD_WS="$REPO_ROOT/rad_workspace"
cd "$RAD_WS" || { echo "ERROR: missing $RAD_WS" >&2; exit 1; }

echo "REPO_ROOT=$REPO_ROOT"
echo "PWD=$(pwd)"


make_series_list() {
  local dir="$1"
  local tag="$2"     # e.g. "3M"
  local M="$3"       # e.g. 3, 6, 30
  local docs_per_file="${4:-500000}"

  if [[ ! -d "$dir" ]]; then
    echo "Directory not found: $dir" >&2
    return 1
  fi

  # Total docs requested and number of files needed
  local total_docs=$(( M * 1000000 ))
  local needed_files=$(( (total_docs + docs_per_file - 1) / docs_per_file ))

  # Collect only part_*.parquet and sort them
  shopt -s nullglob
  local files=( "$dir"/part_*.parquet )
  shopt -u nullglob

  if (( ${#files[@]} == 0 )); then
    echo "No part_*.parquet files found in $dir" >&2
    return 1
  fi

  IFS=$'\n' files=( $(printf '%s\n' "${files[@]}" | sort) )
  unset IFS

  if (( needed_files > ${#files[@]} )); then
    echo "Requested ${needed_files} files but only ${#files[@]} available; using all." >&2
    needed_files=${#files[@]}
  fi

  local parts=()
  for ((i=0; i<needed_files; i++)); do
    local idx=$(( i + 1 ))
    parts+=( "${tag}_${idx}=${files[$i]}" )
  done

  export SERIES_LIST="${parts[*]}"
  echo "SERIES_LIST=${SERIES_LIST}"
}





# #  no cache and with SIMD
# export OUT=/home/nelson/rad_fuzzy_dedup_vectordb/results/30_experiment_cc_main_v2/cc_main_rad_only_v14/
# make_series_list "/home/nelson/rad_fuzzy_dedup_vectordb/data/lm1b_rad_30M" "6M" 100

# ---- DATA ----
export OUT="${OUT:-$REPO_ROOT/results/cc_main_1M_exp/cc_main_1M_results_v2}"
make_series_list "$REPO_ROOT/data/cc_main_1M" "6M" 5




IFS=$'\n\t'



# Columns
ID_COL="${ID_COL:-int_id_column}"
TEXT_COL="${TEXT_COL:-contents}"
MINHASH_COL="${MINHASH_COL:-}"     # leave empty if not available

# Prepare / dedup
QUERIES_N="${QUERIES_N:-1000}"
GLOBAL_SEED="${GLOBAL_SEED:-12345}"
K_PER_DOC="${K_PER_DOC:-112}"
M_BITS="${M_BITS:-4096}"

PERMS_SEED="${PERMS_SEED:-49037}"
MMH3_SEED="${MMH3_SEED:-9173}"
JACCARD_THR="${JACCARD_THR:-0.7}"
NUM_WORKERS="${NUM_WORKERS:-28}"

# Build / query
THREADS="${THREADS:-28}"

# Used by the NEW script as task chunking for MinHash in multiprocessing
MH_BATCH="${MH_BATCH:-2000}"

# Used by the NEW script as build batch size (corpus phase)
ADD_BATCH="${ADD_BATCH:-10000}"

# Probe batch size (query/probe phase)
PROBE_BATCH="${PROBE_BATCH:-10000}"


# efConstruction multiplier knobs (NEW script)
BUILD_EFC_MUL="${BUILD_EFC_MUL:-4.0}"
PROBE_EFC_MUL="${PROBE_EFC_MUL:-4.0}"


# Optional: attribute probe “timed” search+insert throughput to a specific ef (NEW script)
PROBE_TIMING_EF="${PROBE_TIMING_EF:-}"

TOPK="${TOPK:-4}"

# M sets
M_BUILD_LIST="${M_BUILD_LIST:-128}"
M_QUERY_LIST="${M_QUERY_LIST:-128}"


export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# Metrics to run (space separated)
METRICS="${METRICS:-jaccard}"  # e.g., "hamming", "jaccard", or "hamming jaccard"

# Example:
# SERIES_LIST='100K=/data/part_1.parquet 200K=/data/part_2.parquet 300K=/data/part_3.parquet'
SERIES_LIST="${SERIES_LIST:-}"

########################################
# Helpers
########################################

usage() {
  cat << EOF
Usage:
  $(basename "$0") SERIES=PARQUET [SERIES=PARQUET ...]
or:
  SERIES_LIST="100K=/p1 200K=/p2 300K=/p3" ./$(basename "$0")

Notes:
  - OUT is the ROOT folder. Per-series caches/logs go to \$OUT/<SERIES>.
  - Indices go to \$OUT/indices/<SERIES>/<metric>/M<M>.
  - When building series N>1, the script extends from the PREVIOUS series in the ORDER YOU PROVIDE.
EOF
  exit 1
}

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1" >&2; exit 1; }; }

wrapper_for_metric() {
  local metric="$1"
}

# Tokenize whitespace-separated lists into arrays regardless of global IFS.
tokenize_lists() {
  local IFS_BAK="$IFS"; IFS=$' \t\n'
  read -r -a METRICS_ARR   <<< "${METRICS}"
  read -r -a M_BUILD_ARR   <<< "${M_BUILD_LIST}"
  read -r -a M_QUERY_ARR   <<< "${M_QUERY_LIST}"
  IFS="$IFS_BAK"
}

# Find a previous index dir for rad_data_prepare. (use first existing metric×M under previous series)
find_prev_index_dir_for_prepare() {
  local prev_series="$1"
  echo ">>OUT_SERIES>>"
  echo $OUT_SERIES
  echo $prev_series
  echo ">>OUT_SERIES>>"

  local dir
  for metric in "${METRICS_ARR[@]}"; do
    for M in "${M_BUILD_ARR[@]}"; do
      dir="$OUT/${prev_series}/indices/${prev_series}/${metric}/M${M}"
      echo $dir
      if [[ -d "$dir" ]]; then
        echo "$dir"
        return 0
      fi
    done
  done
  return 1
}

########################################
# Parse inputs (order preserved exactly as provided)
########################################

declare -a SERIES_ARR=()
declare -a INPUT_ARR=()

if (( $# > 0 )); then
  # Positional mode: ./run_1m4_append.sh 100K=/p1 200K=/p2 ...
  for pair in "$@"; do
    [[ "$pair" == *=* ]] || { echo "Bad arg: $pair (expect SERIES=PATH)"; usage; }
    SERIES_ARR+=( "${pair%%=*}" )
    INPUT_ARR+=(  "${pair#*=}"  )
  done
elif [[ -n "${SERIES_LIST:-}" ]]; then
  # Env mode: SERIES_LIST='100K=/p1 200K=/p2 ...'  OR  SERIES_LIST=( "100K=/p1" "200K=/p2" )
  if declare -p SERIES_LIST 2>/dev/null | grep -q 'declare \-a'; then
    pairs=( "${SERIES_LIST[@]}" )
  else
    IFS_BAK="$IFS"; IFS=$' \t\n'
    read -r -a pairs <<< "$SERIES_LIST"
    IFS="$IFS_BAK"
  fi
  for pair in "${pairs[@]}"; do
    [[ "$pair" == *=* ]] || { echo "Bad SERIES_LIST token: $pair (expect SERIES=PATH)"; usage; }
    SERIES_ARR+=( "${pair%%=*}" )
    INPUT_ARR+=(  "${pair#*=}"  )
  done
else
  usage
fi

########################################
# Sanity & setup
########################################

need_cmd python
# mkdir -p "$OUT/logs"

# Make sure lists are arrays regardless of IFS
tokenize_lists

echo "== ORDER OF SERIES (as given) =="
for i in "${!SERIES_ARR[@]}"; do
  printf '  %s -> %s\n' "${SERIES_ARR[$i]}" "${INPUT_ARR[$i]}"
done



########################################
# Main loop
########################################

echo $SERIES_ARR

printf '(%s)\n' "$(printf '"%s" ' "${SERIES_ARR[@]}")"

for ((idx=0; idx<${#SERIES_ARR[@]}; idx++)); do
  echo "idx=$idx  val=${SERIES_ARR[idx]}"

  SERIES="${SERIES_ARR[$idx]}"
  INPUT="${INPUT_ARR[$idx]}"

  OUT_SERIES="$OUT/$SERIES"          # per-series cache/log root
  LOG_DIR="$OUT_SERIES/logs"
  mkdir -p "$OUT_SERIES" "$LOG_DIR" "$OUT_SERIES/ground_truth"

  echo "==============================="
  echo ">>> SERIES: $SERIES"
  echo ">>> OUT: $OUT"
  echo ">>> INPUT : $INPUT"
  echo ">>> OUTDIR: $OUT_SERIES (caches/logs) | $OUT (indices/figures)"
  echo "==============================="

  echo $idx


  PREP_PREV_DIR=""
  PREP_PREV_DIR_FILES=""

  if (( idx > 0 )); then
    PREV_SERIES="${SERIES_ARR[$((idx-1))]}"

    # FIXED: missing slash after $OUT
    PREP_PREV_DIR="$OUT/${PREV_SERIES}/manifests/"
    PREP_PREV_DIR_FILES="$OUT/${PREV_SERIES}/cache/"

    if [[ -n "${PREP_PREV_DIR}" ]]; then
      echo "   ↪ rad_data_prepare. will reuse prev_index_dir=${PREP_PREV_DIR}"
      echo "   ↪ rad_data_prepare. will reuse prev_index_dir=${PREP_PREV_DIR_FILES}"
    else
      echo "   ↪ no prev_index_dir found for rad_data_prepare.; proceeding fresh."
    fi
  fi

  echo "== Step 1: rad_data_prepare. ($SERIES) =="
  
  python rad_data_prepare.py \
    --input "$INPUT" \
    --outdir "$OUT_SERIES" \
    --id_col "$ID_COL" \
    --text_col "$TEXT_COL" \
    ${MINHASH_COL:+--minhash_col "$MINHASH_COL"} \
    ${PREP_PREV_DIR:+--prev_index_dir} ${PREP_PREV_DIR:+"$PREP_PREV_DIR"} \
    --queries_n "$QUERIES_N" \
    --global_seed "$GLOBAL_SEED" \
    --K "$K_PER_DOC" --M_bits "$M_BITS" \
    --value_to_bucket mod \
    --perms_seed "$PERMS_SEED" --mmh3_seed "$MMH3_SEED" \
    | tee "$LOG_DIR/01_rad_data_prepare..log"
  echo "✓ Step 1 complete ($SERIES)."


  # ---------- Build + Probe (divergence) ----------
  # Probe parquet default: reuse the query parquet produced by rad_data_prepare
  # Override with: export PROBE_PARQUET=/path/to/queries_100k.parquet
  # PROBE_PARQUET="${PROBE_PARQUET:-$OUT_SERIES/cache/queries_30k.parquet}"

  PROBE_PARQUET=$OUT_SERIES/cache/queries.parquet
  CORPUS_PARQUET=$OUT_SERIES/cache/corpus.parquet

  echo $PROBE_PARQUET

  # Corpus parquet for this series is the input parquet(s)
  # CORPUS_PARQUET="$INPUT"

  for metric in "${METRICS_ARR[@]}"; do
    WRAP="$(wrapper_for_metric "$metric")"

    for M in "${M_QUERY_ARR[@]}"; do
      echo "== Build+Probe divergence : metric=${metric} (M=$M, $SERIES) =="

      # Current-series index paths (same layout as before)
      CUR_DIR="$OUT_SERIES/indices/${SERIES}/${metric}/M${M}"
      CUR_INDEX="$CUR_DIR/index.faiss"
      CUR_LABELS="$CUR_DIR/labels.npy"
      mkdir -p "$CUR_DIR"

      RUN_ARGS=(
        python rad_main.py
        --outdir "$OUT_SERIES"
        --index_path "$CUR_INDEX"
        --labels_npy "$CUR_LABELS"
        --corpus_parquet "$CORPUS_PARQUET"
        --probe_parquet "$PROBE_PARQUET"
        --id_col "$ID_COL" --text_col "$TEXT_COL"
        --threads "$THREADS"
        --mh_task_batch "$MH_BATCH"
        --build_batch "$ADD_BATCH"
        --probe_batch "$PROBE_BATCH"
        --M "$M"
        --topk "$TOPK"
        --build_efC_mul "$BUILD_EFC_MUL"
        --probe_efC_mul "$PROBE_EFC_MUL"
        --series_tag "${SERIES}_M${M}_${metric}_probe_div"
        --IDX "$idx"
      )

      # Optional: choose which ef to attribute probe throughput timing to
      if [[ -n "${PROBE_TIMING_EF}" ]]; then
        RUN_ARGS+=( --probe_timing_ef "$PROBE_TIMING_EF" )
      fi

      # Seed from previous series if available
      if (( idx > 0 )); then
        PREV_SERIES="${SERIES_ARR[$((idx-1))]}"
        PREV_DIR="$OUT/${PREV_SERIES}/indices/${PREV_SERIES}/${metric}/M${M}"
        PREV_INDEX="$PREV_DIR/index.faiss"
        PREV_LABELS="$PREV_DIR/labels.npy"

        if [[ -f "$PREV_INDEX" && -f "$PREV_LABELS" ]]; then
          echo "   ↪ seeding from prev index: $PREV_INDEX"
          RUN_ARGS+=( --prev_index_path "$PREV_INDEX" --prev_labels_npy "$PREV_LABELS" )
        else
          echo "   ↪ prev index not found ($PREV_INDEX); will create NEW index."
        fi
      fi

      # Run (keep wrapper if it is a passthrough wrapper; otherwise run python directly)
      $WRAP "${RUN_ARGS[@]}" | tee "$LOG_DIR/05_build_probe_${metric}_M${M}.log"
      # If wrappers are not compatible, replace the line above with:
      # "${RUN_ARGS[@]}" | tee "$LOG_DIR/05_build_probe_${metric}_M${M}.log"
    done
  done

  echo "✓ Completed SERIES: $SERIES"

done

echo "All series done."
