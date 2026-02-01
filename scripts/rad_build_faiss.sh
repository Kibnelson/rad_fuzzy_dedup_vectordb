#!/usr/bin/env bash
set -euo pipefail

# Repo root = one level up from this script's directory (repo_root/scripts/this_file)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAISS_DIR="${REPO_ROOT}/faiss"
BUILD_DIR="${FAISS_DIR}/build"

JOBS="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)}"

echo "REPO_ROOT = ${REPO_ROOT}"
echo "FAISS_DIR  = ${FAISS_DIR}"
echo "BUILD_DIR  = ${BUILD_DIR}"
echo "JOBS      = ${JOBS}"

cmake -S "${FAISS_DIR}" -B "${BUILD_DIR}" \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=ON \
  -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_C_API=ON -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_OPT_LEVEL=avx512 \
  -DCMAKE_CXX_FLAGS="-mavx512vpopcntdq"

make -C "${BUILD_DIR}" -j "${JOBS}" faiss_avx512
make -C "${BUILD_DIR}" -j "${JOBS}" swigfaiss_avx512

(cd "${BUILD_DIR}/faiss/python" && python setup.py install)