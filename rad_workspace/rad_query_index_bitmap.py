#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Sequence, Tuple, Optional

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import polars as pl
import faiss
import mmh3

import simd_fuzzy_deduplicationv7_unsorted_multi7A_V1 as simd


# ============================================================
# I/O helpers
# ============================================================

def ensure_dir(p: Path) -> None:
    """Create directory (and parents) if needed."""
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path):
    """Read JSON file from path."""
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(p: Path, obj) -> None:
    """Write JSON file with indent=2."""
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_spec_and_perms(outdir: Path):
    """Load sketch spec and permutations used for MinHash."""
    spec = read_json(outdir / "manifests" / "sketch_spec.json")
    perms = np.load(outdir / "manifests" / "permutations.npy")
    return spec, perms


# ============================================================
# MinHash core: blocked multiply–shift kernel
# ============================================================

def _generate_minhash_blocked(
    permutations: np.ndarray,
    text: str,
    window_size: int = 5,
    delimiter: str = " ",
    CH: int = 8192,
) -> np.ndarray:
    """
    Multiply-shift MinHash in memory-bounded blocks.

    permutations: np.uint32[K] or np.int64[K] (cast to uint64)
    returns:      np.uint32[K]
    """
    words = text.split()
    n = len(words)
    K = permutations.shape[0]
    if n == 0:
        return np.zeros((K,), dtype=np.uint32)

    # Build k-shingles with sliding window (ensure ≥1 shingle)
    shingles = [
        delimiter.join(words[i:i + window_size])
        for i in range(max(1, n - window_size + 1))
    ]

    # 64-bit hashes for shingles
    hv = np.array([mmh3.hash(s, signed=False) for s in shingles], dtype=np.uint64)

    a = permutations.astype(np.uint64, copy=False)
    best = np.full((K,), np.uint64((1 << 64) - 1), dtype=np.uint64)

    # Process in CH-sized blocks to bound memory
    for start in range(0, hv.shape[0], CH):
        sl = hv[start:start + CH][:, None]          # (CH, 1)
        vals = (sl * a[None, :]) >> np.uint64(32)   # (CH, K)
        local = vals.min(axis=0)                    # best per permutation in this block
        np.minimum(best, local, out=best)           # in-place min with global best

    return best.astype(np.uint32, copy=False)


# ============================================================
# Worker globals for ProcessPoolExecutor
# ============================================================

_G_PERMS:    np.ndarray | None = None
_G_WINDOW:   int | None = None
_G_CHUNK:    int | None = None
_G_CONTENTS: np.ndarray | None = None  # np.ndarray of str
_G_IDS:      np.ndarray | None = None  # np.ndarray of int

def _init_worker(
    permutations: np.ndarray,
    window_size: int,
    block_CH: int,
    contents: np.ndarray,
    ids: np.ndarray,
):
    """
    Initializer for worker processes.
    Stores big arrays and config once per process (COW-friendly).
    """
    global _G_PERMS, _G_WINDOW, _G_CHUNK, _G_CONTENTS, _G_IDS
    _G_PERMS    = permutations
    _G_WINDOW   = int(window_size)
    _G_CHUNK    = int(block_CH)
    _G_CONTENTS = contents
    _G_IDS      = ids

def _minhash_worker_fast_range(start: int, end: int) -> List[Tuple[int, np.ndarray]]:
    """
    Worker for range [start:end): compute MinHash and return (doc_id, mh) pairs.
    """
    assert _G_PERMS is not None
    assert _G_WINDOW is not None
    assert _G_CONTENTS is not None
    assert _G_IDS is not None
    out: List[Tuple[int, np.ndarray]] = []

    perms = _G_PERMS
    wsize = _G_WINDOW
    CH    = _G_CHUNK or 8192
    C     = _G_CONTENTS
    I     = _G_IDS

    for idx in range(start, end):
        mh = _generate_minhash_blocked(perms, C[idx], window_size=wsize, CH=CH)
        out.append((int(I[idx]), mh))
    return out


# ============================================================
# Parallel MinHash driver (even split across workers)
# ============================================================

def parallel_minhash_fast(
    df_pd: pd.DataFrame,
    permutations: np.ndarray,
    *,
    window_size: int,
    n_proc: int,
    block_CH: int = 8192,
    batch_size: int,
    verbose: bool = True,
    return_items: bool = False,  # if True: return [{"minhashes":..., "int_id_column":...}, ...]
    contents_col: str = "contents",
    id_col: str = "int_id_column",
) -> List[Tuple[int, np.ndarray]] | List[Dict[str, Any]]:
    """
    Compute MinHash for each row using blocked multiply-shift, in parallel.

    Work is split evenly across `n_proc` workers:

      - Each worker gets a contiguous [start:end) slice.
      - Big arrays (contents, ids, perms) live in globals (COW under fork).

    Returns by default:
        [(doc_id: int, mh: np.ndarray[uint32, K]), ...] sorted by doc_id.

    If return_items=True, returns:
        [{"minhashes": mh, "int_id_column": doc_id}, ...]
    """
    N = len(df_pd)
    if N == 0:
        return []

    # Extract columns once; on Linux these are COW under fork.
    contents = df_pd[contents_col].values
    ids      = df_pd[id_col].values

    workers = n_proc


     # Build index ranges for tasks (avoid pickling large arrays)
    bs = max(1, batch_size)
    ranges = [(off, min(off + bs, N)) for off in range(0, N, bs)]
    num_batches = len(ranges)

    if verbose:
        print(
            f"[parallel_minhash_fast] rows={N} workers={workers} "
            f"batches={num_batches}"
        )

    t0 = time.perf_counter()
    results: List[Tuple[int, np.ndarray]] = []

    # Single-process path (keeps behavior identical to worker function)
    if workers == 1 or num_batches == 1:
        _init_worker(permutations, window_size, block_CH, contents, ids)
        for (s, e) in ranges:
            results.extend(_minhash_worker_fast_range(s, e))
    else:
        # Multi-process path
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(permutations, int(window_size), int(block_CH), contents, ids),
        ) as pool:
            fut_meta: Dict[Any, Tuple[int, int, int, float]] = {}
            for i, (s, e) in enumerate(ranges, 1):
                t_submit = time.perf_counter()
                fut = pool.submit(_minhash_worker_fast_range, s, e)
                fut_meta[fut] = (i, s, e, t_submit)

            done = 0
            for fut in as_completed(fut_meta):
                i, s, e, t_submit = fut_meta[fut]
                try:
                    chunk = fut.result()
                except Exception as ex:
                    raise RuntimeError(
                        f"_minhash_worker_fast_range failed for [{s}:{e})"
                    ) from ex
                results.extend(chunk)
                done += 1
                if verbose:
                    print(
                        f"[parallel_minhash_fast] done {done}/{num_batches} "
                        f"range=[{s}:{e}) wait={(time.perf_counter()-t_submit):.3f}s"
                    )

    total_s = time.perf_counter() - t0
    if verbose:
        print(
            f"[parallel_minhash_fast] total={total_s:.3f}s "
            f"({total_s/60:.2f} min), items={len(results)}"
        )

    # Stable order by doc_id
    results.sort(key=lambda t: t[0])

    if return_items:
        return [
            {"minhashes": mh, "int_id_column": doc_id}
            for (doc_id, mh) in results
        ]
    return results


def to_items_tuples(tuples_list: list[tuple[int, np.ndarray]]) -> list[dict]:
    """
    Convert [(doc_id, mh)] -> [{"minhashes":..., "int_id_column":...}, ...]
    (keeps compatibility with older code paths).
    """
    return [
        {"minhashes": mh.astype(np.uint32, copy=False), "int_id_column": doc_id}
        for (doc_id, mh) in tuples_list
    ]


# ============================================================
# Polars streaming
# ============================================================

def _rows_iter_pl(ldf: pl.LazyFrame, total: int, batch_rows: int):
    """Yield small Polars DataFrames by slicing a LazyFrame in row chunks."""
    start = 0
    while start < total:
        stop = min(start + batch_rows, total)
        yield ldf.slice(start, stop - start).collect(engine="streaming")
        start = stop


# ============================================================
# MinHash → bitmap vectorization
# ============================================================

def minhash_to_bitmap_mmh3(
    minhashes: Sequence[int],
    M: int = 4096,
    seed: int = 0,
    endian: str = "little",
) -> np.ndarray:
    """
    Map a MinHash vector to an M-bit bitmap via mmh3.hash64.

    returns: np.uint64[(M+63)//64]
    """
    words = np.zeros((M + 63) // 64, dtype=np.uint64)
    pow2 = (M & (M - 1)) == 0
    mask = M - 1 if pow2 else None
    for v in minhashes:
        x = int(v) & 0xFFFFFFFF
        b = x.to_bytes(4, endian)
        h = mmh3.hash64(b, seed=seed, signed=False)[0]
        idx = (h & mask) if pow2 else (h % M)
        words[idx >> 6] |= (np.uint64(1) << np.uint64(idx & 63))
    return words

def minhashes_to_vectors(
    mh_uint64: np.ndarray,
    vectorizer: str,
    M_bits: int,
    mmh3_seed: int,
) -> np.ndarray:
    """
    Convert an array of MinHash rows into FAISS-ready binary vectors.

    mh_uint64: np.ndarray [N, K] of uint32/uint64 hashes
    returns:   np.uint8 [N, M_bits/8] bit-packed
    """
    N, _K = mh_uint64.shape
    words_list = [
        minhash_to_bitmap_mmh3(
            mh_uint64[i, :].tolist(),
            M=M_bits,
            seed=mmh3_seed,
        ).view(np.uint8)
        for i in range(N)
    ]
    X = np.vstack([w.reshape(1, -1) for w in words_list])
    return X


# ============================================================
# Ground truth / evaluation helpers
# ============================================================

def load_ground_truth(gt_path: Optional[Path]) -> Optional[Dict[int, List[int]]]:
    """Load ground truth JSON mapping qid -> list of GT ids."""
    if not gt_path:
        return None
    gt = read_json(gt_path)
    return {int(k): v for k, v in gt.items()}

def compute_metric(
    I: np.ndarray,
    q_ids: List[int],
    gt: Dict[int, List[int]],
    k: int,
):
    """
    Compute either hit@k (when each GT has ≤1 id) or recall@k (multi-GT case).
    Returns (score, mode, n_counted, successes_or_None).
    """
    if not gt:
        return None, "none", 0, None

    # Case 1: each query has at most one GT id → hit@k
    single = all(len(v) <= 1 for v in gt.values())
    if single:
        hits = 0
        counted = 0
        for qi, qid in enumerate(q_ids):
            g = gt.get(qid)
            if not g:
                continue
            counted += 1
            true_id = g[0] if g else None
            if true_id is not None and true_id in I[qi, :k]:
                hits += 1
        return ((hits / counted) if counted else None), "hit@k", counted, hits

    # Case 2: multi-GT: compute recall@k
    acc = 0.0
    counted = 0
    for qi, qid in enumerate(q_ids):
        g = gt.get(qid)
        if not g:
            continue
        denom = min(k, len(g))
        if denom == 0:
            continue
        acc += len(set(I[qi, :k].tolist()) & set(g[:denom])) / float(denom)
        counted += 1
    return ((acc / counted) if counted else None), "recall@k", counted, None


# ============================================================
# HNSW parameter scheduling by chunk index
# ============================================================

def get_hnsw_params_for_chunk(
    chunk_idx: int,
    M: int,
    *,
    # How many 500K-chunks per "phase"
    chunks_per_phase: int = 5,

    # efConstruction = (efc_base_mul + phase * efc_step_mul) * M
    efc_base_mul: int = 7,
    efc_step_mul: int = -1,
    efc_max_mul: Optional[int] = None,

    # efSearch = efsearch_base + phase * efsearch_step
    efsearch_base: int = 600,
    efsearch_step: int = -50,
    efsearch_max: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Compute (efConstruction, efSearch) for a given 500K-chunk index.

    chunk_idx       : 1-based index of the current 500K chunk (1..N).
    M               : HNSW M (max neighbors per node).
    chunks_per_phase: number of chunks in each phase.
    efc_base_mul    : base multiplier for efConstruction (in units of M).
    efc_step_mul    : per-phase increment for the multiplier.
    efc_max_mul     : optional max multiplier for efConstruction.
    efsearch_base   : base efSearch for phase 0.
    efsearch_step   : per-phase increment for efSearch.
    efsearch_max    : optional max efSearch.
    """
    if chunk_idx < 1:
        raise ValueError("chunk_idx must be ≥ 1 (1-based index)")
    if chunks_per_phase <= 0:
        raise ValueError("chunks_per_phase must be ≥ 1")

    # 0-based phase index: 0 for chunks 1..chunks_per_phase,
    #                      1 for next group, etc.
    phase = (chunk_idx - 1) // chunks_per_phase

    # efConstruction multiplier
    efc_mul = efc_base_mul + phase * efc_step_mul
    if efc_max_mul is not None:
        efc_mul = min(efc_mul, efc_max_mul)
    efC = efc_mul * M

    # efSearch value
    efSearch = efsearch_base + phase * efsearch_step
    if efsearch_max is not None:
        efSearch = min(efSearch, efsearch_max)

    return efC, efSearch


# ============================================================
# Main pipeline: MinHash → SIMD in-batch dedup → HNSW search
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        "query_index_simple_simd_vector "
        "(SIMD in-batch dedup + FAISS; parallel MinHash like simd_inbatch_dedup_nocorpus.py)"
    )
    ap.add_argument("--outdir", required=True, type=Path)

    # Index & data
    ap.add_argument(
        "--index_path",
        required=True,
        type=Path,
        help="Path to FAISS binary index (bare IndexBinaryHNSW)",
    )
    ap.add_argument(
        "--labels_npy",
        type=Path,
        default=None,
        help="Path to labels.npy; default: sibling of index_path",
    )
    ap.add_argument(
        "--queries_parquet",
        type=Path,
        help="Default: out/cache/queries_30k.parquet",
    )
    ap.add_argument("--id_col", default="int_id_column")
    ap.add_argument("--text_col", default="contents")

    # Sketch & pipeline params
    ap.add_argument(
        "--vectorizer",
        choices=["bitmap", "minhash"],
        default="bitmap",
        help="bitmap=MinHash→bitmap; minhash=raw MinHash 8-byte words",
    )
    ap.add_argument("--window", type=int, default=5)

    # Unified thread knob
    ap.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 8)),
        help="Number of worker processes / SIMD threads / FAISS OMP threads",
    )

    # Kept for backward compatibility (no longer used)
    ap.add_argument(
        "--mh_batch",
        type=int,
        default=10_000,
        help="(deprecated, unused) kept for backward compatibility",
    )
    ap.add_argument(
        "--query_batch",
        type=int,
        default=10_000,
        help="queries per FAISS batch",
    )
    ap.add_argument(
        "--n_proc",
        type=int,
        default=None,
        help="(deprecated, ignored) use --threads instead",
    )

    # SIMD dedup params
    ap.add_argument("--simd_threshold", type=float, default=0.7)
    ap.add_argument("--simd_bands", type=int, default=14)
    ap.add_argument(
        "--simd_threads",
        type=int,
        default=None,
        help="(deprecated, ignored) use --threads instead",
    )

    ap.add_argument("--ef_list", type=int, nargs="+", default=[400])
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--IDX", required=True, type=int, default=None)

    # Ground truth (optional)
    ap.add_argument(
        "--gt_json",
        type=Path,
        default=None,
        help="Optional ground truth JSON mapping qid -> GT ids",
    )

    # Outputs
    ap.add_argument(
        "--series_tag",
        default="M48_hamming_simd_inbatch",
        help="used in output filename; e.g., M48_hamming_simd_inbatch",
    )
    ap.add_argument(
        "--out_metrics",
        type=Path,
        default=None,
        help="Explicit output path; default under out/metrics/recall_curves/<series_tag>.json",
    )
    ap.add_argument(
        "--save_neighbors",
        action="store_true",
        help="If set, save per-ef neighbor lists to parquet files",
    )
    ap.add_argument(
        "--neighbors_dir",
        type=Path,
        default=None,
        help="Directory to write neighbor files; default under out/results/neighbors/<series_tag>/",
    )
    ap.add_argument(
        "--save_dedup_drops",
        action="store_true",
        help="If set, save per-batch dropped query ids as parquet",
    )

    args = ap.parse_args()

    outdir = args.outdir.resolve()
    cdir = outdir / "cache"
    ensure_dir(outdir / "metrics" / "recall_curves")

    # Normalize threads
    threads = max(1, int(args.threads) if args.threads is not None else (os.cpu_count() or 8))
    print(f"Using threads={threads} for MinHash + SIMD + FAISS")

    # Load spec & permutations (frozen in Step 1)
    spec, perms = load_spec_and_perms(outdir)
    M_bits  = int(spec["M_bits"])
    mmh3_sd = int(spec["mmh3_seed"])
    d_bits  = int(spec["M_bits"])
    efC = int(args.M) * 4

    print("efC Chosen:", efC)

    # Load FAISS index + labels
    faiss.omp_set_num_threads(threads)
    index = faiss.read_index_binary(str(args.index_path))

    # Debug: show existing efSearch/efConstruction
    print("efSearch:", index.hnsw.efSearch)
    print("efConstruction:", index.hnsw.efConstruction)

    if hasattr(index, "index") and hasattr(index.index, "hnsw"):
        index = index.index  # intentionally kept commented as in original

    if not hasattr(index, "hnsw"):
        raise RuntimeError(
            "Index is not bare IndexBinaryHNSW (no .hnsw). Rebuild with Option B."
        )

    labels_path = args.labels_npy or (args.index_path.parent / "labels.npy")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.npy not found: {labels_path}")
    labels = np.load(labels_path).astype(np.int64, copy=False)
    if index.ntotal != labels.shape[0]:
        raise RuntimeError("labels.npy length mismatch with index.ntotal")

    # Load queries as LazyFrame
    qparq = args.queries_parquet or (cdir / "queries_30k.parquet")
    if not qparq.exists():
        raise FileNotFoundError(f"Queries parquet not found: {qparq}")
    q_ldf = (
        pl.scan_parquet(qparq)
          .select(
              pl.col(args.id_col).alias("int_id_column"),
              pl.col(args.text_col).alias("contents"),
          )
          .with_columns(
              pl.col("int_id_column").cast(pl.Int64),
              pl.col("contents").cast(pl.Utf8),
          )
    )
    q_total = q_ldf.select(pl.len()).collect(engine="streaming").item()
    print(f"• Queries: {q_total:,} from {qparq}")

    # Prepare outputs
    series_tag = args.series_tag
    out_metrics = args.out_metrics or (
        outdir / "metrics" / "recall_curves" / f"{series_tag}.json"
    )
    neighbors_dir = args.neighbors_dir or (
        outdir / "results" / "neighbors" / series_tag
    )
    if args.save_neighbors:
        ensure_dir(neighbors_dir)
    if args.save_dedup_drops:
        ensure_dir(outdir / "results" / "dedup_drops" / series_tag)

    gt = load_ground_truth(args.gt_json)
    k = args.topk
    
    results = {
        "series": series_tag,
        "vectorizer": args.vectorizer,
        "topk": k,
        "ef_list": args.ef_list,
        "points": [],
    }

    # ========================================================
    # Sweep over efSearch values
    # ========================================================
    for ef in args.ef_list:
        # Dup counters (for reporting)
        DUP_THRESHOLD = 0.3  # normalized in [0, 1]
        n_inserted_total = 0
        n_duplicates_total = 0

        DUP_FRAC = 0.3
        DUP_THRESHOLD_BITS = int(DUP_FRAC)  # kept for parity, though distance is normalized

        n_inserted_total = 0
        n_duplicates_total = 0
        known_ids = set(labels.tolist())

        # Debug existing parms
        print("efSearch:", index.hnsw.efSearch)
        print("efConstruction:", index.hnsw.efConstruction)

        # Existing: set HNSW ef params on .hnsw
        index.hnsw.efConstruction = int(efC)
        index.hnsw.efSearch = int(ef)

        # Debug verify
        print("efSearch:", index.hnsw.efSearch)
        print("efConstruction:", index.hnsw.efConstruction)

        # Per-batch stats
        per_batch_avg_ms: List[float] = []
        per_batch_query_ms: List[float] = []
        per_insert_batch_dt_ms: List[float] = []
        per_batch_dt_s: List[float] = []
        per_batch_sizes: List[int] = []
        per_batch_dedup_ms: List[float] = []
        per_batch_minhash_ms: List[float] = []

        per_batch_dedup_size: List[int] = []
        all_Ipos: List[np.ndarray] = []
        q_ids_kept_all: List[int] = []

        per_batch_total_s: List[float] = []
        per_batch_total_all_ms: List[float] = []
        per_batch_total_no_minhash_ms: List[float] = []

        CH = args.query_batch
        t0_all = time.time()
        start_idx = 0

        keept_ids_list: List[int] = []
        remove_ids_list: List[int] = []

        # ================================================
        # Batch loop: MinHash → SIMD dedup → search → insert
        # ================================================
        for df_pl in _rows_iter_pl(q_ldf, q_total, CH):

            # Existing: set HNSW ef params on .hnsw
            index.hnsw.efConstruction = int(efC)
            index.hnsw.efSearch = int(ef)

            # New: also set top-level ef (kept for compatibility / experimentation)
            index.efConstruction = int(efC)
            index.efSearch = int(ef)

            t0_a = time.time()

            # Normalize column names to expected ones
            df_pd = (
                df_pl
                .rename({args.id_col: "int_id_column", args.text_col: "contents"})
                .to_pandas()
            )
            df_pd = df_pd.sort_values("int_id_column").reset_index(drop=True)

            # 1) Parallel MinHash (collect timing)
            t0_d = time.time()
            items_tuples = parallel_minhash_fast(
                df_pd,
                perms,
                window_size=args.window,
                n_proc=threads,
                block_CH=8192,
                batch_size=args.mh_batch,
                verbose=False,
            )
            minhash_dt_ms = (time.time() - t0_d) * 1000.0
            per_batch_minhash_ms.append(minhash_dt_ms)
            print(f"[MinHash helper] rows={len(df_pd)} mh={minhash_dt_ms:.1f}ms (wall {minhash_dt_ms:.3f} s)")

            # Convert to dict format used by SIMD dedup path
            items = to_items_tuples(items_tuples)
            if not items:
                start_idx += len(df_pd)
                continue

            # 2) SIMD in-batch dedup (on MinHash matrix)
            items.sort(key=lambda d: d["int_id_column"])
            mh_mat = np.asarray(
                [it["minhashes"] for it in items],
                dtype=np.uint64,
            )

            t0_d = time.time()
            nested = simd.simd_fuzzy_deduplicationv7_unsorted_multi7A_V1(
                mh_mat,
                threshold=args.simd_threshold,
                num_bands=args.simd_bands,
                num_threads=threads,
            )
            dedup_dt_ms = (time.time() - t0_d) * 1000.0
            per_batch_dedup_ms.append(dedup_dt_ms)

            # Flatten nested structure of indices to a removal set
            to_remove = set()
            if nested is not None:
                for x in nested:
                    if isinstance(x, (list, tuple, np.ndarray)):
                        for y in x:
                            if isinstance(y, (list, tuple, np.ndarray)):
                                for z in y:
                                    to_remove.add(int(z))
                            else:
                                to_remove.add(int(y))
                    else:
                        to_remove.add(int(x))

            keep_idx = [i for i in range(mh_mat.shape[0]) if i not in to_remove]
            per_batch_dedup_size.append(len(items))
            keept_ids_list.append(len(keep_idx))
            remove_ids_list.append(len(to_remove))
            if not keep_idx:
                start_idx += len(df_pd)
                continue
            
            kept_ids = [int(items[i]["int_id_column"]) for i in keep_idx]
            kept_mh  = mh_mat[keep_idx, :]
            q_ids_kept_all.extend(kept_ids)

            # 3) Convert kept MinHashes to bitmap vectors
            XQ = minhashes_to_vectors(kept_mh, args.vectorizer, M_bits, mmh3_sd)

            # Warmup search: trigger pb warm-build with minimal workload
            t0 = time.time()
            if XQ.shape[0] >= 1:
                _D_warm, _I_warm = index.search(XQ[:2], 1)
            dt_ms = (time.time() - t0) * 1000.0

            # 4) FAISS search (collect timing)
            t0 = time.time()
            D, Ipos = index.search(XQ, k)  # positions into labels (0..N-1)
            dt_ms = (time.time() - t0) * 1000.0
            per_batch_query_ms.append(dt_ms)
            per_batch_avg_ms.append(dt_ms / max(1, XQ.shape[0]))

            # Distance-based duplicate mask (normalized [0,1])
            scale = 100.0
            threshold = 0.3
            thr_int = int(round(threshold * scale))  # 0.3 -> 30


            if D.size:
                min_dists_bits = D.min(axis=1)
                dup_mask = (min_dists_bits <= 30)
            else:
                dup_mask = np.zeros((XQ.shape[0],), dtype=bool)

            # Keep running duplicate/insert stats
            n_dup_batch  = int(np.sum(dup_mask))
            n_keep_batch = int(np.sum(~dup_mask))
            n_duplicates_total += n_dup_batch
            dt_Insertms = 0.0

            if n_keep_batch > 0:
                Xcand = XQ[~dup_mask, :]
                ids_cand = np.asarray(kept_ids, dtype=np.int64)[~dup_mask]

                # Avoid re-inserting existing IDs
                new_mask = np.array(
                    [iid not in known_ids for iid in ids_cand],
                    dtype=bool,
                )

                if new_mask.any():
                    # Existing: set HNSW params on .hnsw
                    index.hnsw.efConstruction = int(efC)
                    index.hnsw.efSearch = 10

                    # New: top-level fields (no-op for IndexBinaryHNSW but kept)
                    index.efConstruction = int(efC)
                    index.efSearch = 10

                    XInsert = Xcand[new_mask, :]
                    ids_to_insert = ids_cand[new_mask]

                    # 5) Insert back into HNSW (collect timing)
                    tInsert = time.time()
                    index.add(XInsert)
                    dt_Insertms = (time.time() - tInsert) * 1000.0
                    per_insert_batch_dt_ms.append(dt_Insertms)

                    labels = np.concatenate([labels, ids_to_insert])
                    known_ids.update(ids_to_insert.tolist())
                    n_inserted_total += XInsert.shape[0]

                    # Persist index + labels (inside batch loop)
                    faiss.write_index_binary(index, str(args.index_path))
                    np.save(labels_path, labels.astype(np.int64, copy=False))

            # Per-batch totals (ms)
            batch_total_ms = (time.time() - t0_a) * 1000.0
            per_batch_total_s.append(batch_total_ms)
            per_batch_sizes.append(int(XQ.shape[0]))
            all_Ipos.append(Ipos)

            # Aggregate timing with and without MinHash
            per_batch_total_no_minhash_ms.append(
                (dedup_dt_ms + dt_ms + dt_Insertms)
            )
            per_batch_total_all_ms.append(
                (minhash_dt_ms + dedup_dt_ms + dt_ms + dt_Insertms)
            )

            # Persist index + labels after batch
            faiss.write_index_binary(index, str(args.index_path))
            np.save(labels_path, labels.astype(np.int64, copy=False))


            # args.save_neighbors = True
            # args.save_dedup_drops = True
    
            if args.save_neighbors:
                ensure_dir(neighbors_dir)
            if args.save_dedup_drops:
                ensure_dir(outdir / "results" / "dedup_drops" / series_tag)

            if args.save_neighbors:
                df = pl.DataFrame({
                    "query_id": kept_ids,
                    "neighbors": [
                        labels[Ipos[i, :k]].tolist()
                        for i in range(Ipos.shape[0])
                    ],
                })
                out_pq = neighbors_dir / f"ef{ef}_start{start_idx}.parquet"
                df.write_parquet(out_pq, compression="zstd")

            if args.save_dedup_drops:
                original_ids_order = [
                    int(it["int_id_column"]) for it in items
                ]
                dropped_ids = [
                    original_ids_order[i]
                    for i in range(len(items))
                    if i not in set(keep_idx)
                ]
                if dropped_ids:
                    df_drop = pl.DataFrame({
                        "batch_start": [start_idx] * len(dropped_ids),
                        "dropped_query_id": dropped_ids,
                    })
                    out_drop = (
                        outdir / "results" / "dedup_drops" / series_tag /
                        f"ef{ef}_start{start_idx}.parquet"
                    )
                    df_drop.write_parquet(out_drop, compression="zstd")

            start_idx += len(df_pd)

        # Final persist after all batches for this ef
        faiss.write_index_binary(index, str(args.index_path))
        np.save(labels_path, labels.astype(np.int64, copy=False))

        # ===========================
        # Metrics aggregation
        # ===========================
        if len(all_Ipos) == 0:
            Ipos = np.zeros((0, k), dtype=np.int64)
            elapsed_s = time.time() - t0_all
            score = None
            mode = "none"
            counted = 0
            successes = None
        else:
            Ipos = np.vstack(all_Ipos)
            elapsed_s = time.time() - t0_all
            I = np.where(Ipos >= 0, labels[Ipos], -1)
            score = None
            mode = "none"
            counted = 0
            successes = None
            if gt is not None:
                score, mode, counted, successes = compute_metric(
                    I, q_ids_kept_all, gt, k
                )

        # MinHash stats
        minhash_total_ms = float(np.sum(per_batch_minhash_ms))
        minhash_avg_ms_per_query = (
            minhash_total_ms /
            max(1, int(np.sum(per_batch_dedup_size)))
        )
        minhash_p50_ms = (
            float(np.percentile(per_batch_minhash_ms, 50))
            if per_batch_minhash_ms
            else 0.0
        )

        # Dedup stats
        dedup_total_ms = float(np.sum(per_batch_dedup_ms))
        dedup_avg_ms_per_query = (
            dedup_total_ms /
            max(1, int(np.sum(per_batch_dedup_size)))
        )
        dedup_p50_ms = (
            float(np.percentile(per_batch_dedup_ms, 50))
            if per_batch_dedup_ms
            else 0.0
        )

        # Search stats (FAISS only)
        total_dt_ms = float(np.sum(per_batch_query_ms))
        n_total = int(np.sum(per_batch_sizes))
        avg_ms = (total_dt_ms / max(1, n_total)) if n_total else 0.0
        p50_ms = (
            float(np.percentile(per_batch_avg_ms, 50))
            if per_batch_avg_ms
            else 0.0
        )
        p95_ms = (
            float(np.percentile(per_batch_avg_ms, 95))
            if per_batch_avg_ms
            else 0.0
        )
        p99_ms = (
            float(np.percentile(per_batch_avg_ms, 99))
            if per_batch_avg_ms
            else 0.0
        )
        min_ms = (
            float(np.min(per_batch_avg_ms))
            if per_batch_avg_ms
            else 0.0
        )
        max_ms = (
            float(np.max(per_batch_avg_ms))
            if per_batch_avg_ms
            else 0.0
        )

        # Insert stats
        total_dt_insert_ms = float(np.sum(per_insert_batch_dt_ms))
        n_total_inserts = int(np.sum(per_batch_sizes))
        avg_insert_ms = (
            total_dt_insert_ms / max(1, n_total_inserts)
        ) if n_total_inserts else 0.0

        total_processing_s = float(np.sum(per_batch_total_s))
        total_processing_all_s = float(np.sum(per_batch_total_all_ms))
        total_processing_no_minhash_s = float(
            np.sum(per_batch_total_no_minhash_ms)
        )

        sizes = np.asarray(per_batch_sizes, dtype=float)
        tot_s = np.asarray(per_batch_total_no_minhash_ms, dtype=float) / 1000.0

        tot_s_list = np.asarray(per_batch_total_s, dtype=float).tolist()
        per_batch_dt_ms_list = np.asarray(per_batch_query_ms, dtype=float).tolist()
        per_insert_batch_dt_s_list = np.asarray(per_insert_batch_dt_ms, dtype=float).tolist()
        per_batch_minhash_ms_list = np.asarray(per_batch_minhash_ms, dtype=float).tolist()
        sizes_list = np.asarray(per_batch_sizes, dtype=float).tolist()

        results["points"].append({
            "ef": ef,
            "metric": mode,
            "remove_ids_list": remove_ids_list,
            "keept_ids_list": keept_ids_list,

            "new_total_all_batch": tot_s_list,
            "per_batch_dt_ms_only_query": per_batch_dt_ms_list,
            "per_insert_batch_dt_s_list": per_insert_batch_dt_s_list,

            "per_batch_minhash_ms_list": per_batch_minhash_ms_list,
            "sizes_list": sizes_list,
            "total_processing_all_s": total_processing_all_s,
            "total_processing_no_minhash_s": total_processing_no_minhash_s,
            "total_queries": int(np.sum(per_batch_sizes)),

            "throughput_all_steps": int(np.sum(per_batch_sizes)) / (total_processing_all_s / 1000),
            "throughput_all_steps_no_minhash": int(np.sum(per_batch_sizes)) / (total_processing_no_minhash_s / 1000),

            "score": score,
            "n_counted": counted,
            "successes": (int(successes) if successes is not None else None),
            "n_queries": int(np.sum(per_batch_sizes)),
            "search_total_ms": total_dt_ms,
            "insert_total_ms": total_dt_insert_ms,

            "dedup_total_ms": dedup_total_ms,
            "minhash_total_ms": minhash_total_ms,
            "total_processing_s": total_processing_s,
            "n_duplicates_dropped": int(n_duplicates_total),
            "n_inserted_back": int(n_inserted_total),
        })

        print(
            f"  ef={ef:<3d} {mode}={score if score is not None else 'NA'} "
            f"keptQ={int(np.sum(per_batch_sizes))} "
            f"dedup_total={dedup_total_ms:.1f}ms "
            f"search_avg={avg_ms:.2f}ms p95={p95_ms:.2f}ms"
        )

    write_json(out_metrics, results)
    print(f" Wrote metrics → {out_metrics}")


if __name__ == "__main__":
    main()
