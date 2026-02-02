# RAD

Research workspace for **RAD: Online Fuzzy Deduplication of Evolving Datasets via Approximate Nearest Neighbor Search**: online fuzzy deduplication for evolving corpora using **HNSW retrieval** over compact document signatures (FAISS) plus classic **DPK-style MinHash/LSH** pipelines for ground truth and comparisons.

This repository is a **meta-repo** that pins two upstream projects as submodules:

- **FAISS** (`faiss/`) — fork of `facebookresearch/faiss`, used for HNSW / binary indexing experiments and RAD’s bitmap/Jaccard distance kernels.
- **Data Prep Kit (DPK)** (`data-prep-kit/`) — fork of `data-prep-kit/data-prep-kit`, used for data preparation and DPK-based ground-truth generation.

> **Note on large data:** This repo does **not** store large datasets in Git (e.g., large Parquet shards). Use the provided download utilities or external storage. See **Data & large files**.

---

## Paper → code map (what to review)

RAD’s end-to-end pipeline matches the paper workflow (Figure 3):

1. **Signature generation**: shingle → MinHash → pack into **RAD bitmap signatures**
2. **In-batch dedup**: classic MinHash/LSH-style filtering inside the incoming batch (SIMD-accelerated)
3. **Index search**: query the **HNSW** index over bitmap signatures
4. **Candidate filter**: threshold using **Jaccard-aligned** distances (default `J > 0.7`)
5. **Insert**: write unique docs to disk and insert their signatures into the index

Where these live in the repo:

- **Pipeline orchestration (Python + shell):** `rad_workspace/`
- **SIMD acceleration for DPK-style steps:** `dpk_simd/` (optional compilation)
- **FAISS HNSW + distance kernels (C++):** `faiss/` submodule fork

---

## Repository layout (high-level)

- `faiss/` — FAISS submodule (fork; RAD/HNSW distance code lives here)
- `data-prep-kit/` — Data Prep Kit submodule (fork; ground truth + reference pipeline)
- `rad_workspace/` — main research code (pipelines + Python scripts)
- `scripts/` — machine + environment setup helpers
- `dpk_simd/` — SIMD extension / helpers for DPK-style operations (optional compilation)
- `data/` — small test data only (no large shards)
- `results/` — experiment outputs (generated)

---

## Tested environment

- OS: Ubuntu **noble minimal** (image built 2026-01-29)
- Arch: x86_64 / amd64
- CPU: AMD Genoa (AVX-512)

---

## Clone (with submodules)

```bash
sudo apt update
sudo apt install git -y

git clone --recurse-submodules https://github.com/Kibnelson/rad_fuzzy_dedup_vectordb.git
cd rad_fuzzy_dedup_vectordb
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

---

## Machine setup (Ubuntu noble)

Install system dependencies + Docker setup:

```bash
sudo bash scripts/setup_ubuntu_noble.sh --set-python-alternative --docker-nonroot
```

Optional: install AWS CLI v2 too:

```bash
sudo bash scripts/setup_ubuntu_noble.sh --with-awscli --set-python-alternative --docker-nonroot
```

Verify:

```bash
python3.10 --version
docker --version
```

---

## Build / activate the project environment (run from repo root)

### 1) Build the DPK `fdedup` virtualenv (one-time)

```bash
cd data-prep-kit-rad/transforms/universal/dpk_fdedup
make venv
cd ../../../..
```

### 2) Activate the project runtime env

From the **repo root**:

```bash
source scripts/setup_env.sh
pip install -r scripts/requirements.txt
```

This script:
- detects the repo root automatically,
- activates the `fdedup` venv,
- sets `PYTHONPATH` for `data-prep-kit`, `rad_workspace`, and `dpk_simd`.

> Important: use `source ...` (not `bash ...`) so the environment stays active in your current shell.

### 3) Optional: compile SIMD module (`dpk_simd/`)

`dpk_simd/` contains the SIMD-accelerated helpers used to speed up DPK-style hot loops (band processing / candidate intersection).

```bash
bash dpk_simd/compile.sh
```

Skip this step, since we have added a precompiled but if you are running with different configurations you may have to compile.

---

## Build FAISS (RAD HNSW + distance kernels)

With the env activated, build FAISS from the repo root:

```bash
chmod +x scripts/rad_build_faiss.sh
./scripts/rad_build_faiss.sh
```

### Where our FAISS changes live (to review)

Our FAISS modifications are contained in the forked `faiss/` submodule (branch `research-rad`). Key implementation points:

- `faiss/IndexBinaryHNSW.cpp`
- `faiss/utils/hamming_distance/avx512-inl.h`

These files contain the binary/HNSW distance and SIMD-related changes used by our FAISS baselines and RAD kernels.

---

## Data & large files

### Important rule
Do **not** commit large datasets (e.g., big `.parquet` shards) to GitHub. Use one of:
- the download utilities in `rad_workspace/utilities/`,
- cloud storage (GCS/S3),
- a local path outside the repo.

### Dataset download utilities

All download scripts live under:

- `rad_workspace/utilities/`

Examples:

```bash
python rad_workspace/utilities/download_c4_data.py
python rad_workspace/utilities/download_common_crawl_data_v2.py
python rad_workspace/utilities/download_lm1b_data.py
python rad_workspace/utilities/download_realnews_data.py
```

(Each script writes into a local `data/` subfolder or a configured path—check the script header/args.)

---

## Test data (default paths)

This repo includes a small **test dataset** and the executor scripts are pre-configured to run against it by default.

In the executor scripts, the default dataset + output directory are set in the `# ---- DATA ----` block, e.g.:

```bash
# ---- DATA ----
export OUT="${OUT:-$REPO_ROOT/results/cc_main_1M_exp/cc_main_1M_results_v1}"
make_series_list "$REPO_ROOT/data/cc_main_1M" "6M" 5
```

---

## Running experiments

All pipelines are designed to be launched from the **repo root** (after `source scripts/setup_env.sh`).

### A) Full pipeline with ground truth (DPK-based)

This pipeline:
- prepares data
- deduplicates corpus + queries
- builds indices
- computes ground truth using DPK
- queries the index and reports recall/throughput

Run:

```bash
bash rad_workspace/run_dpk_ground_truth_pipeline_executor.sh
```

### B) Pipeline without ground truth

This pipeline runs build/query without recall evaluation:

```bash
bash rad_workspace/rad_no_ground_truth_pipeline_executor.sh
```

### C) Milvus approximate pipeline (vector DB baseline)

This pipeline uses **Milvus** as the ANN backend (Approximate retrieval).

#### 1) Start Milvus (Docker Compose)

From the **repo root** (recommended):

```bash
# Make the launcher executable (one-time)
chmod +x rad_workspace/milvus/start_milvus_dockerv2.sh

# Start Milvus (runs docker compose, waits until ready)
rad_workspace/milvus/start_milvus_dockerv2.sh
```

From **any folder** (pick your `rad_workspace` explicitly):

```bash
/path/to/rad_fuzzy_dedup_vectordb/rad_workspace/milvus/start_milvus_dockerv2.sh   --ws /path/to/rad_fuzzy_dedup_vectordb/rad_workspace
```

**Note:** The default Milvus launcher clears persistent volumes under `rad_workspace/milvus/` (etcd + MinIO + Milvus data) for a clean start. If you want persistence across runs, disable the “wipe volumes” block in `start_milvus_dockerv2.sh`.

#### 2) Run the Milvus approximate executor

Once Milvus is up:

```bash
chmod +x rad_workspace/run_milvus_approximate.sh
bash rad_workspace/run_milvus_approximate.sh
```

Override the output directory if needed:

```bash
export OUT="$REPO_ROOT/results/milvus_approx_run_1"
bash rad_workspace/run_milvus_approximate.sh
```

### Outputs

Experiment outputs are written under:

- `results/<experiment_name>/...`

Most runs log into per-series folders under the chosen `OUT` directory.

---

## Notes on series runs and `OUT`

Some executor scripts run multi-stage “series” experiments (e.g., `6M_1`, `6M_2`, ...), extending indices from previous series.

Common pattern:
- `OUT` is the experiment root under `results/`
- per-series outputs go under `$OUT/<SERIES>/...`

Override `OUT` before running:

```bash
export OUT="/path/to/results/your_experiment"
bash rad_workspace/run_dpk_ground_truth_pipeline_executor.sh
```

---

## Contributing / syncing with upstream

Add upstream remotes **inside each submodule**.

### FAISS upstream

```bash
cd faiss
git remote add upstream https://github.com/facebookresearch/faiss.git
git remote -v
cd ..
```

### Data Prep Kit upstream

```bash
cd data-prep-kit
git remote add upstream https://github.com/data-prep-kit/data-prep-kit.git
git remote -v
cd ..
```

---

## Troubleshooting

### “Environment didn’t activate”
You must run:

```bash
source scripts/setup_env.sh
```

Running `bash scripts/setup_env.sh` starts a child shell and exits—your current shell will not keep the env changes.

### “Module not found” (rad_workspace / dpk paths)
Ensure:
- you sourced `scripts/setup_env.sh`
- you are launching pipelines from repo root

### “Large Parquet push rejected”
GitHub rejects files > 100MB (unless configured for LFS, and even then large data is better kept out of Git).
Use download scripts or external storage instead.
