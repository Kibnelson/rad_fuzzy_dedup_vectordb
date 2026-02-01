#!/usr/bin/env bash
set -e

# ---- find repo root (git first, else walk up for .git) ----
START_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REPO_ROOT="$(git -C "$START_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  d="$START_DIR"
  while [[ "$d" != "/" ]]; do
    if [[ -d "$d/.git" ]]; then
      REPO_ROOT="$d"
      break
    fi
    d="$(dirname "$d")"
  done
fi

if [[ -z "$REPO_ROOT" ]]; then
  echo "ERROR: could not detect repo root from $START_DIR" >&2
  exit 1
fi

export REPO_ROOT
cd "$REPO_ROOT" || exit 1
echo "REPO_ROOT=$REPO_ROOT"
echo "PWD=$(pwd)"

# ---- venv ----
source "$REPO_ROOT/data-prep-kit/transforms/universal/fdedup/venv/bin/activate"
export PATH="$REPO_ROOT/data-prep-kit/transforms/universal/fdedup/venv/bin:$PATH"

# ---- PYTHONPATH (safe even if previously unset) ----
SITEPKG="$(python -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$SITEPKG:$REPO_ROOT/data-prep-kit/data-processing-lib/python/src:$REPO_ROOT/data-prep-kit/data-processing-lib/spark/src:$REPO_ROOT/data-prep-kit/data-processing-lib/ray/src:$REPO_ROOT/data-prep-kit/transforms/universal/fdedup:$REPO_ROOT/data-prep-kit/transforms/universal/fdedup/dpk_fdedup:$REPO_ROOT/data-prep-kit/transforms/universal/fdedup/dpk_fdedup/spark:$REPO_ROOT/data-prep-kit/transforms/universal/fdedup/dpk_fdedup/ray:$REPO_ROOT/data-prep-kit/transforms/universal/filter/dpk_filter:$REPO_ROOT/data-prep-kit/transforms/universal/filter/dpk_filter/ray:$REPO_ROOT/data-prep-kit/transforms/universal/filter/dpk_filter/spark:$REPO_ROOT/dpk_simd:${PYTHONPATH:-}"
