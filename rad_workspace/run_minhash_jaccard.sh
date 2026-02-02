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





# ---- DATA ----
export OUT="${OUT:-$REPO_ROOT/results/cc_main_1M_exp/cc_main_1M_results_v1}"
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
# M_BITS="${M_BITS:-4096}"
M_BITS="${M_BITS:-3584}"



PERMS_SEED="${PERMS_SEED:-49037}"
MMH3_SEED="${MMH3_SEED:-9173}"
JACCARD_THR="${JACCARD_THR:-0.7}"
NUM_WORKERS="${NUM_WORKERS:-28}"

# Build / query
THREADS="${THREADS:-28}"
MH_BATCH="${MH_BATCH:-2000}"
ADD_BATCH="${ADD_BATCH:-10000}"
EFC="${EFC:-300}"

TOPK="${TOPK:-4}"


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
  if [[ "$metric" == "hamming" ]]; then echo "./run_hamming_s3.sh"
  elif [[ "$metric" == "jaccard" ]]; then echo "./run_jaccard_s3.sh"
  else echo ""; fi
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

# for ((idx=0; idx<${#SERIES_ARR[@]}; idx++)); do
for ((idx=0; idx<${#SERIES_ARR[@]}; idx++)); do
  echo "idx=$idx  val=${SERIES_ARR[idx]}"

# for idx in "${!SERIES_ARR[@]}"; do
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
    PREP_PREV_DIR="$OUT${PREV_SERIES}/manifests/"
    PREP_PREV_DIR_FILES="$OUT${PREV_SERIES}/cache/"

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

  echo "== Step 2: rad_data_dedup_corpus (corpus) ($SERIES) =="
  python rad_data_dedup_corpus.py \
    --outdir "$OUT_SERIES" \
    --id_col "$ID_COL" \
    --text_col "$TEXT_COL" \
    --corpus_cache "$OUT_SERIES/cache/corpus_970k.parquet" \
    --num_workers "$NUM_WORKERS" \
    --jaccard_threshold "$JACCARD_THR" \
    | tee "$LOG_DIR/02_rad_data_dedup_corpus_corpus.log"
  echo "✓ Step 2 (corpus) complete ($SERIES)."

  echo "== Step 2A: rad_data_dedup_corpus (queries) ($SERIES) =="
  python rad_data_dedup_corpus.py \
    --outdir "$OUT_SERIES" \
    --id_col "$ID_COL" \
    --text_col "$TEXT_COL" \
    --corpus_cache "$OUT_SERIES/cache/queries_30k.parquet" \
    --num_workers "$NUM_WORKERS" \
    --jaccard_threshold "$JACCARD_THR" \
    | tee "$LOG_DIR/02A_rad_data_dedup_corpus_queries.log"
  echo "✓ Step 2A (queries) complete ($SERIES)."


  # ---------- Build indices (extend previous when possible) ----------
  for metric in "${METRICS_ARR[@]}"; do
  
    for M in "${M_BUILD_ARR[@]}"; do
      echo "== rad_build_index_minhash $metric (M=$M, $SERIES) =="

      BUILD_ARGS=(
        python rad_build_index_minhash.py
        --outdir "$OUT_SERIES"                     # indices under $OUT/indices/<SERIES>/<metric>/M<M>
        --corpus_parquet "$OUT_SERIES/cache/corpus_970k_dedup.parquet"
        --id_col "$ID_COL" --text_col "$TEXT_COL"
        --series "$SERIES" --metric "$metric" --M "$M"
        --efC "$EFC" --threads "$THREADS" --mh_batch "$MH_BATCH" --add_batch "$ADD_BATCH"
      )

      echo ">>>>>>>>>IDX = ($idx)"

      if (( idx > 0 )); then
       
        PREV_SERIES="${SERIES_ARR[$((idx-1))]}"
        # PREV_DIR="$OUT_SERIES/indices/${PREV_SERIES}/${metric}/M${M}"
        PREV_DIR="$OUT/${PREV_SERIES}/indices/${PREV_SERIES}/${metric}/M${M}"

        echo ">>>>>>>>>PREV_DIR = ($PREV_DIR)"

        if [[ -d "$PREV_DIR" ]]; then
          echo "   ↪ extending from prev_index_dir=$PREV_DIR"
          BUILD_ARGS+=( --prev_index_dir "$PREV_DIR" )

          printf '%q ' "${BUILD_ARGS[@]}"; echo

        else
          echo "   ↪ prev_index_dir not found ($PREV_DIR); building fresh."
        fi
      
      fi

       "${BUILD_ARGS[@]}" | tee "$LOG_DIR/04_build_${metric}_M${M}.log"
    done
  done

  # xxx

  echo "== Step 3: rad_prepare_ground_truth ($SERIES) =="
  python rad_prepare_ground_truth.py \
    --outdir "$OUT_SERIES" \
    --queries_parquet "$OUT_SERIES/cache/queries_30k_dedup.parquet" \
    --corpus_parquet  "$OUT_SERIES/cache/corpus_970k_dedup.parquet" \
    --id_col "$ID_COL" --text_col "$TEXT_COL" \
    --gt_k 1 \
    ${PREP_PREV_DIR_FILES:+--prev_index_dir} ${PREP_PREV_DIR_FILES:+"$PREP_PREV_DIR_FILES"} \
    --out_json "$OUT_SERIES/ground_truth/gt_top1_${SERIES}.json" \
    --num_workers "$NUM_WORKERS" --jaccard_threshold "$JACCARD_THR" \
    | tee "$LOG_DIR/03_rad_prepare_ground_truth.log"
  


  #     --save_neighbors \
  # ---------- Query indices ----------
  for metric in "${METRICS_ARR[@]}"; do
    for M in "${M_QUERY_ARR[@]}"; do
      echo "== rad_query_index_minhash : Bitmap→${metric^} (M=$M, $SERIES) =="
     python rad_query_index_minhash.py \
        --outdir "$OUT_SERIES" \
        --index_path   "$OUT_SERIES/indices/${SERIES}/${metric}/M${M}/index.faiss" \
        --labels_npy   "$OUT_SERIES/indices/${SERIES}/${metric}/M${M}/labels.npy" \
        --queries_parquet "$OUT_SERIES/cache/queries_30k.parquet" \
        --id_col "$ID_COL" --text_col "$TEXT_COL" \
        --series_tag "M${M}_${metric}" \
        --gt_json "$OUT_SERIES/ground_truth/gt_top1_${SERIES}.json" \
        --threads "$THREADS" --simd_threads "$THREADS" --n_proc "$THREADS" \
        --query_batch 10000 --mh_batch "$MH_BATCH" \
        --M "$M" \
        --IDX "$idx" \
        --topk "$TOPK" \
        --neighbors_dir "$OUT_SERIES/results/neighbors/${SERIES}_M${M}_${metric}" \
        | tee "$LOG_DIR/05_query_${metric}_M${M}.log"
    done
  done

  # xxx
  echo "✓ Completed SERIES: $SERIES"
  

done

echo "🎉 All series done."
