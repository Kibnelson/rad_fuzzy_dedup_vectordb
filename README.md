# rad_fuzzy_dedup_vectordb

Research workspace for **RAD fuzzy deduplication + vector database** experiments (FAISS + Data Prep Kit).

This repository is a **meta-repo** that pins two upstream projects as submodules:

- **FAISS** (fork of `facebookresearch/faiss`) — used for HNSW / binary indexing experiments
- **Data Prep Kit (DPK)** (fork of `data-prep-kit/data-prep-kit`) — used for data preparation + ground-truth generation

> **Note on large data:** This repo does **not** store large datasets in Git (e.g., large Parquet shards). Use the provided download utilities or external storage. See **Data & large files**.

---

## Tested environment

- OS: Ubuntu **noble minimal** (image built 2026-01-29)
- Arch: x86_64 / amd64
- CPU: AMD Genoa (AVX-512)

---

## Repository layout (high-level)

- `faiss/` — FAISS submodule (fork)
- `data-prep-kit/` — Data Prep Kit submodule (fork)
- `rad_workspace/` — main research code (pipelines + Python scripts live here)
- `scripts/` — machine + environment setup helpers
- `dpk_simd/` — SIMD extension / helpers (optional compilation)
- `data/` — small test data (only small files; no large shards)
- `results/` — experiment outputs (generated)

If your clone uses a different submodule directory layout, ensure you have `faiss/` and `data-prep-kit/` at repo root (or adjust paths in scripts accordingly).

---

## Clone (with submodules)

```bash
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
cd data-prep-kit/transforms/universal/fdedup
make venv
cd ../../../..
```

### 2) Activate the project runtime env

From the **repo root**:

```bash
source scripts/setup_env.sh
```

This script:
- detects the repo root automatically,
- activates the `fdedup` venv,
- sets `PYTHONPATH` for `data-prep-kit`, `rad_workspace`, and `dpk_simd`.

> Important: use `source ...` (not `bash ...`) so the environment stays active in your current shell.

### 3) Optional: compile SIMD module

If you need to rebuild the SIMD extension:

```bash
bash dpk_simd/compile.sh
```

---

## Build FAISS (RAD HNSW with Bitmap-Jaccard)

With the env activated, build FAISS:

```bash
in Then from the repo root (rad_fuzzy_dedup_vectordb) you can run:
chmod +x scripts/rad_build_faiss.sh
./scripts/rad_build_faiss.sh
```

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

## Running experiments

All pipelines are designed to be launched from the **repo root** (after `source scripts/setup_env.sh`).

### A) Full pipeline with ground truth (DPK-based)

This pipeline:
- prepares data
- deduplicates corpus + queries
- builds indices
- computes ground truth using DPK
- queries the index and reports recall

Run:

```bash
bash rad_workspace/run_dpk_ground_truth_pipeline_executor.sh
```

### B) Pipeline without ground truth

This pipeline runs build/query without recall evaluation:

```bash
bash rad_workspace/rad_no_ground_truth_pipeline_executor.sh
```

### Outputs

Experiment outputs are written under:

- `results/<experiment_name>/...`

Most runs log into per-series folders under the chosen `OUT` directory.

---

## Notes on series runs and `OUT`

Some executor scripts build multi-stage “series” runs (e.g., `6M_1`, `6M_2`, ...), extending indices from previous series.

Common pattern:
- `OUT` is the experiment root under `results/`
- per-series outputs go under `$OUT/<SERIES>/...`

If needed, override `OUT` before running:

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
