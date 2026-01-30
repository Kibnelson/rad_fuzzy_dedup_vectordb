#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(p: Path, obj) -> None:
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_spec_and_perms(outdir: Path):
    spec = read_json(outdir / "manifests" / "sketch_spec.json")
    perms = np.load(outdir / "manifests" / "permutations.npy")
    return spec, perms


# ============================================================
# MinHash: blocked multiply–shift kernel
# ============================================================

def _generate_minhash_blocked(
    permutations: np.ndarray,
    text: str,
    window_size: int = 5,
    delimiter: str = " ",
    CH: int = 8192,
) -> np.ndarray:
    words = text.split()
    n = len(words)
    K = permutations.shape[0]
    if n == 0:
        return np.zeros((K,), dtype=np.uint32)

    shingles = [
        delimiter.join(words[i:i + window_size])
        for i in range(max(1, n - window_size + 1))
    ]
    hv = np.array([mmh3.hash(s, signed=False) for s in shingles], dtype=np.uint64)

    a = permutations.astype(np.uint64, copy=False)
    best = np.full((K,), np.uint64((1 << 64) - 1), dtype=np.uint64)

    for start in range(0, hv.shape[0], CH):
        sl = hv[start:start + CH][:, None]          # (CH, 1)
        vals = (sl * a[None, :]) >> np.uint64(32)   # (CH, K)
        local = vals.min(axis=0)
        np.minimum(best, local, out=best)

    return best.astype(np.uint32, copy=False)


# ============================================================
# Worker globals for ProcessPoolExecutor
# ============================================================

_G_PERMS:    np.ndarray | None = None
_G_WINDOW:   int | None = None
_G_CHUNK:    int | None = None
_G_CONTENTS: np.ndarray | None = None
_G_IDS:      np.ndarray | None = None

def _init_worker(
    permutations: np.ndarray,
    window_size: int,
    block_CH: int,
    contents: np.ndarray,
    ids: np.ndarray,
):
    global _G_PERMS, _G_WINDOW, _G_CHUNK, _G_CONTENTS, _G_IDS
    _G_PERMS    = permutations
    _G_WINDOW   = int(window_size)
    _G_CHUNK    = int(block_CH)
    _G_CONTENTS = contents
    _G_IDS      = ids

def _minhash_worker_fast_range(start: int, end: int) -> List[Tuple[int, np.ndarray]]:
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

def parallel_minhash_fast(
    df_pd: pd.DataFrame,
    permutations: np.ndarray,
    *,
    window_size: int,
    n_proc: int,
    block_CH: int = 8192,
    batch_size: int = 20000,
    verbose: bool = False,
    contents_col: str = "contents",
    id_col: str = "int_id_column",
) -> List[Tuple[int, np.ndarray]]:
    N = len(df_pd)
    if N == 0:
        return []

    contents = df_pd[contents_col].values
    ids      = df_pd[id_col].values
    workers = max(1, min(int(n_proc), N))

    bs = max(1, int(batch_size))
    ranges = [(off, min(off + bs, N)) for off in range(0, N, bs)]

    if verbose:
        print(f"[parallel_minhash_fast] rows={N} workers={workers} batches={len(ranges)}")

    t0 = time.perf_counter()
    results: List[Tuple[int, np.ndarray]] = []

    if workers == 1 or len(ranges) == 1:
        _init_worker(permutations, window_size, block_CH, contents, ids)
        for (s, e) in ranges:
            results.extend(_minhash_worker_fast_range(s, e))
    else:
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

            for fut in as_completed(fut_meta):
                _i, s, e, _t_submit = fut_meta[fut]
                chunk = fut.result()
                results.extend(chunk)

    if verbose:
        total_s = time.perf_counter() - t0
        print(f"[parallel_minhash_fast] total={total_s:.3f}s items={len(results)}")

    results.sort(key=lambda t: t[0])
    return results

def to_items_tuples(tuples_list: List[Tuple[int, np.ndarray]]) -> List[Dict[str, Any]]:
    return [
        {"minhashes": mh.astype(np.uint32, copy=False), "int_id_column": doc_id}
        for (doc_id, mh) in tuples_list
    ]


# ============================================================
# Polars streaming
# ============================================================

def _rows_iter_pl(ldf: pl.LazyFrame, total: int, batch_rows: int):
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
    M_bits: int,
    mmh3_seed: int,
) -> np.ndarray:
    N, _K = mh_uint64.shape
    words_list = [
        minhash_to_bitmap_mmh3(mh_uint64[i, :].tolist(), M=M_bits, seed=mmh3_seed).view(np.uint8)
        for i in range(N)
    ]
    X = np.vstack([w.reshape(1, -1) for w in words_list])
    return X


# ============================================================
# Index + labels loader / seeding
# ============================================================

def load_or_seed_index_labels(
    *,
    spec: Dict[str, Any],
    index_path: Path,
    labels_path: Path,
    M_hnsw: int,
    efC: int,
    prev_index_path: Optional[Path],
    prev_labels_path: Optional[Path],
    omp_threads: int,
) -> Tuple[faiss.IndexBinaryHNSW, np.ndarray]:
    faiss.omp_set_num_threads(int(omp_threads))
    ensure_dir(index_path.parent)

    have_target = index_path.exists() and labels_path.exists()

    if not have_target and prev_index_path is not None:
        if prev_labels_path is None:
            prev_labels_path = prev_index_path.parent / "labels.npy"
        if not prev_index_path.exists() or not prev_labels_path.exists():
            raise FileNotFoundError(f"prev index/labels missing: {prev_index_path} / {prev_labels_path}")

        print(f"• Seeding from PREVIOUS index: {prev_index_path}")
        index = faiss.read_index_binary(str(prev_index_path))
        labels = np.load(prev_labels_path).astype(np.int64, copy=False)

        if not hasattr(index, "hnsw"):
            raise RuntimeError("Previous index is not bare IndexBinaryHNSW (no .hnsw).")
        if index.ntotal != labels.shape[0]:
            raise RuntimeError("Previous labels.npy length mismatch with index.ntotal")

        faiss.write_index_binary(index, str(index_path))
        np.save(labels_path, labels.astype(np.int64, copy=False))
        have_target = True

    if have_target:
        index = faiss.read_index_binary(str(index_path))
        if not hasattr(index, "hnsw"):
            raise RuntimeError("Index is not bare IndexBinaryHNSW (no .hnsw).")
        labels = np.load(labels_path).astype(np.int64, copy=False)
        if index.ntotal != labels.shape[0]:
            raise RuntimeError("labels.npy length mismatch with index.ntotal")
        print(f"• Loaded CURRENT index: {index_path} (ntotal={index.ntotal:,})")
        return index, labels

    print("• No existing index found; creating NEW IndexBinaryHNSW")
    d_bits = int(spec["M_bits"])
    index = faiss.IndexBinaryHNSW(int(d_bits), int(M_hnsw))
    index.hnsw.efConstruction = int(efC)
    index.hnsw.efSearch = 32
    labels = np.empty((0,), dtype=np.int64)
    faiss.write_index_binary(index, str(index_path))
    np.save(labels_path, labels.astype(np.int64, copy=False))
    print(f"• Created index: {index_path} (ntotal={index.ntotal:,})")
    return index, labels


# ============================================================
# Divergence (order-free) metrics for 2+ EFs, especially 3
# ============================================================

def _ids_to_sets(I_ids: np.ndarray) -> List[set]:
    # I_ids: [N, K] -> list of sets length N
    # remove -1 (missing)
    return [set(row[row >= 0].tolist()) for row in I_ids]

def divergence_order_free_summary(
    by_ef_ids: Dict[int, np.ndarray],  # ef -> [N, K] of IDs
    topk: int,
) -> Dict[str, Any]:
    efs = sorted(by_ef_ids.keys())
    N = next(iter(by_ef_ids.values())).shape[0]
    K = int(topk)

    sets_by_ef = {ef: _ids_to_sets(arr) for ef, arr in by_ef_ids.items()}

    # Pairwise overlap@K and Jaccard
    pairwise = []
    for i in range(len(efs)):
        for j in range(i + 1, len(efs)):
            a, b = efs[i], efs[j]
            overlaps = []
            jaccs = []
            for qi in range(N):
                Sa = sets_by_ef[a][qi]
                Sb = sets_by_ef[b][qi]
                inter = len(Sa & Sb)
                union = len(Sa | Sb)
                overlaps.append(inter / float(K) if K else 0.0)
                jaccs.append((inter / float(union)) if union else 1.0)
            pairwise.append({
                "ef_a": a,
                "ef_b": b,
                "avg_overlap_at_k": float(np.mean(overlaps)) if overlaps else 0.0,
                "avg_jaccard": float(np.mean(jaccs)) if jaccs else 0.0,
            })

    # Frequency bucketization (works for 2+ EFs; for 3 EFs it gives all/2/1)
    frac_all = []
    frac_exactly2 = []
    frac_exactly1 = []
    per_query_counts = []

    for qi in range(N):
        freq: Dict[int, int] = {}
        for ef in efs:
            for x in sets_by_ef[ef][qi]:
                freq[x] = freq.get(x, 0) + 1

        c_all = sum(1 for _x, c in freq.items() if c == len(efs))
        c_2   = sum(1 for _x, c in freq.items() if c == 2) if len(efs) >= 3 else 0
        c_1   = sum(1 for _x, c in freq.items() if c == 1)

        frac_all.append(c_all / float(K) if K else 0.0)
        frac_exactly2.append(c_2 / float(K) if K else 0.0)
        frac_exactly1.append(c_1 / float(K) if K else 0.0)

        per_query_counts.append({
            "count_all": c_all,
            "count_exactly2": c_2,
            "count_exactly1": c_1,
        })

    return {
        "efs": efs,
        "n_queries_compared": int(N),
        "topk": int(K),
        "pairwise": pairwise,
        "avg_frac_common_to_all": float(np.mean(frac_all)) if frac_all else 0.0,
        "avg_frac_in_exactly2": float(np.mean(frac_exactly2)) if frac_exactly2 else 0.0,
        "avg_frac_unique_to_one": float(np.mean(frac_exactly1)) if frac_exactly1 else 0.0,
        # keep per-query counts if you want to verify later (can be large; keep on by default)
        "per_query_freq_counts": per_query_counts,
    }


# ============================================================
# Main pipeline: BUILD then PROBE (divergence), no GT
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        "rad_exp_simd_inbatch_build_and_probe_divergence (no GT)"
    )
    ap.add_argument("--outdir", required=True, type=Path)

    # Index paths
    ap.add_argument("--index_path", required=True, type=Path)
    ap.add_argument("--labels_npy", required=True, type=Path)
    ap.add_argument("--prev_index_path", type=Path, default=None)
    ap.add_argument("--prev_labels_npy", type=Path, default=None)

    # Data
    ap.add_argument("--corpus_parquet", required=True, type=Path)
    ap.add_argument("--probe_parquet", required=True, type=Path)
    ap.add_argument("--id_col", default="int_id_column")
    ap.add_argument("--text_col", default="contents")

    # MinHash / SIMD
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8)))
    ap.add_argument("--mh_task_batch", type=int, default=20000)   # matches your bash
    ap.add_argument("--simd_threshold", type=float, default=0.7)
    ap.add_argument("--simd_bands", type=int, default=14)

    # Build/probe batching
    ap.add_argument("--build_batch", type=int, default=100000)
    ap.add_argument("--probe_batch", type=int, default=10000)

    # HNSW
    ap.add_argument("--M", type=int, default=192)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--ef_list", type=int, nargs="+", default=[200, 300, 400])

    ap.add_argument("--build_efC_mul", type=float, default=2.0)   # matches bash
    ap.add_argument("--probe_efC_mul", type=float, default=1.0)   # matches bash

    # Duplicate filter threshold (Hamming distance in bits)
    ap.add_argument("--dup_bits_threshold", type=int, default=30)

    # Throughput timing attribution during probe
    ap.add_argument("--probe_timing_ef", type=int, default=None)  # if None => base ef (first in ef_list)

    # Metadata / output
    ap.add_argument("--series_tag", default="probe_div")
    ap.add_argument("--IDX", type=int, default=None)  # accept for compatibility
    ap.add_argument("--save_probe_ids_npz", action="store_true")  # optional: save ef->IDs arrays to npz

    args = ap.parse_args()

    outdir = args.outdir.resolve()
    ensure_dir(outdir / "metrics" / "probe_divergence")

    spec, perms = load_spec_and_perms(outdir)
    M_bits  = int(spec["M_bits"])
    mmh3_sd = int(spec["mmh3_seed"])

    threads = max(1, int(args.threads))
    faiss.omp_set_num_threads(int(threads))

    ef_list = list(map(int, args.ef_list))
    if len(ef_list) < 2:
        raise ValueError("--ef_list must contain at least 2 values (e.g., 200 300)")
    base_ef = ef_list[0]
    base_ef = 300
    timing_ef = int(args.probe_timing_ef) if args.probe_timing_ef is not None else base_ef
    timing_ef = 300
    if timing_ef not in ef_list:
        raise ValueError(f"--probe_timing_ef={timing_ef} must be one of ef_list={ef_list}")

    build_efC = int(round(float(args.build_efC_mul) * float(args.M)))
    probe_efC = int(round(float(args.probe_efC_mul) * float(args.M)))

    print(f"Using threads={threads}")
    print(f"M={args.M} topk={args.topk}")
    print(f"ef_list={ef_list} base_ef={base_ef} probe_timing_ef={timing_ef}")
    print(f"build_efC={build_efC} (mul={args.build_efC_mul}) probe_efC={probe_efC} (mul={args.probe_efC_mul})")
    print(f"dup_bits_threshold={args.dup_bits_threshold}")

    # Load/seed index+labels
    index, labels = load_or_seed_index_labels(
        spec=spec,
        index_path=args.index_path,
        labels_path=args.labels_npy,
        M_hnsw=int(args.M),
        efC=int(build_efC),
        prev_index_path=args.prev_index_path,
        prev_labels_path=args.prev_labels_npy,
        omp_threads=int(threads),
    )

    # Track known IDs
    known_ids = set(labels.tolist())

    # Data lazyframes
    def scan_parquet(pq: Path) -> Tuple[pl.LazyFrame, int]:
        ldf = (
            pl.scan_parquet(pq)
              .select(
                  pl.col(args.id_col).alias("int_id_column"),
                  pl.col(args.text_col).alias("contents"),
              )
              .with_columns(
                  pl.col("int_id_column").cast(pl.Int64),
                  pl.col("contents").cast(pl.Utf8),
              )
        )
        total = int(ldf.select(pl.len()).collect(engine="streaming").item())
        return ldf, total

    corpus_ldf, corpus_total = scan_parquet(args.corpus_parquet)
    probe_ldf, probe_total = scan_parquet(args.probe_parquet)

    print(f"• Corpus: {corpus_total:,} from {args.corpus_parquet}")
    print(f"• Probe:  {probe_total:,} from {args.probe_parquet}")

    # --------------------------
    # BUILD phase
    # --------------------------
    build: Dict[str, Any] = {
        "phase": "build",
        "batch_size": int(args.build_batch),
        "dup_bits_threshold": int(args.dup_bits_threshold),
        "build_efC": int(build_efC),
        "build_efSearch": int(base_ef),  # build uses base ef for duplicate filtering (except first empty batch)
        "batch_starts": [],
        "batch_raw": [],
        "batch_kept_simd": [],
        "batch_simd_rm": [],
        "batch_oob_searched": [],
        "batch_oob_dist_dups": [],
        "batch_oob_survivors_after_dist": [],
        "batch_knownid_dups": [],
        "batch_inserted": [],
        "batch_ms_minhash": [],
        "batch_ms_simd": [],
        "batch_ms_search": [],
        "batch_ms_insert": [],
        "batch_ms_end2end": [],
    }

    build_wall_t0 = time.time()

    for start_idx, df_pl in enumerate(_rows_iter_pl(corpus_ldf, corpus_total, int(args.build_batch))):
        batch_start = int(sum(build["batch_raw"])) if build["batch_raw"] else 0

        # normalize / pandas
        df_pd = df_pl.to_pandas().sort_values("int_id_column").reset_index(drop=True)
        raw_n = int(len(df_pd))
        if raw_n == 0:
            continue

        t_batch0 = time.time()

        # MinHash
        t0 = time.time()
        items_tuples = parallel_minhash_fast(
            df_pd,
            perms,
            window_size=int(args.window),
            n_proc=int(threads),
            block_CH=8192,
            batch_size=int(args.mh_task_batch),
            verbose=False,
        )
        mh_ms = (time.time() - t0) * 1000.0

        items = to_items_tuples(items_tuples)
        items.sort(key=lambda d: d["int_id_column"])
        mh_mat = np.asarray([it["minhashes"] for it in items], dtype=np.uint64)

        # SIMD in-batch
        t0 = time.time()
        nested = simd.simd_fuzzy_deduplicationv7_unsorted_multi7A_V1(
            mh_mat,
            threshold=float(args.simd_threshold),
            num_bands=int(args.simd_bands),
            num_threads=int(threads),
        )
        simd_ms = (time.time() - t0) * 1000.0

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
        kept_simd = int(len(keep_idx))
        simd_rm = int(len(to_remove))

        if kept_simd == 0:
            # record & continue
            build["batch_starts"].append(batch_start)
            build["batch_raw"].append(raw_n)
            build["batch_kept_simd"].append(0)
            build["batch_simd_rm"].append(simd_rm)
            build["batch_oob_searched"].append(0)
            build["batch_oob_dist_dups"].append(0)
            build["batch_oob_survivors_after_dist"].append(0)
            build["batch_knownid_dups"].append(0)
            build["batch_inserted"].append(0)
            build["batch_ms_minhash"].append(float(mh_ms))
            build["batch_ms_simd"].append(float(simd_ms))
            build["batch_ms_search"].append(0.0)
            build["batch_ms_insert"].append(0.0)
            build["batch_ms_end2end"].append(float((time.time() - t_batch0) * 1000.0))
            continue

        kept_ids = np.array([int(items[i]["int_id_column"]) for i in keep_idx], dtype=np.int64)
        kept_mh  = mh_mat[keep_idx, :]

        XQ = minhashes_to_vectors(kept_mh, int(M_bits), int(mmh3_sd))

        # BUILD: if index is empty (first batch in brand-new series), DO NOT search;
        # insert all SIMD survivors (after known-id filter) as you requested.
        oob_searched = 0
        search_ms = 0.0
        dup_mask_dist = np.zeros((XQ.shape[0],), dtype=bool)

        if index.ntotal > 0:
            index.hnsw.efConstruction = int(build_efC)
            index.hnsw.efSearch = int(base_ef)

            
            index.efConstruction = int(build_efC)
            index.efSearch = int(base_ef)


            # Warmup search: trigger pb warm-build with minimal workload
            t0 = time.time()
            if XQ.shape[0] >= 1:
                _D_warm, _I_warm = index.search(XQ[:2], 1)
            dt_ms = (time.time() - t0) * 1000.0
            print("==========warm dt_ms==========")
            print(dt_ms)
            


            t0 = time.time()
            D, Ipos = index.search(XQ, int(args.topk))
            search_ms = (time.time() - t0) * 1000.0
            oob_searched = int(XQ.shape[0])

            # Distance dup filter
            if D.size:
                min_d = D.min(axis=1)
                dup_mask_dist = (min_d <= int(args.dup_bits_threshold))
            else:
                dup_mask_dist = np.zeros((XQ.shape[0],), dtype=bool)
        else:
            D = np.empty((XQ.shape[0], int(args.topk)), dtype=np.float32)
            Ipos = np.full((XQ.shape[0], int(args.topk)), -1, dtype=np.int64)

        oob_dist_dups = int(np.sum(dup_mask_dist))
        survivors_after_dist = int(np.sum(~dup_mask_dist))

        inserted = 0
        knownid_dups = 0
        insert_ms = 0.0

        print("survivors_after_dist>>")
        print(survivors_after_dist)

        if survivors_after_dist > 0:
            ids_cand = kept_ids[~dup_mask_dist]
            Xcand = XQ[~dup_mask_dist, :]

            # Avoid re-inserting existing IDs
            new_mask = np.array([int(iid) not in known_ids for iid in ids_cand], dtype=bool)
            knownid_dups = int(np.sum(~new_mask))

            if new_mask.any():
                print("DOING INSERT>>")
                XInsert = Xcand[new_mask, :]
                ids_to_insert = ids_cand[new_mask]

                index.hnsw.efConstruction = int(build_efC)
                # efSearch during add isn’t used the same way, but keep small & stable
                index.hnsw.efSearch = 32


                # New: top-level fields (no-op for IndexBinaryHNSW but kept)
                index.efConstruction = int(build_efC)
                index.efSearch = 32

                t0 = time.time()
                index.add(XInsert)
                insert_ms = (time.time() - t0) * 1000.0

                # Append labels
                labels = np.concatenate([labels, ids_to_insert.astype(np.int64, copy=False)])
                known_ids.update(map(int, ids_to_insert.tolist()))
                inserted = int(XInsert.shape[0])

                # Persist
                faiss.write_index_binary(index, str(args.index_path))
                np.save(args.labels_npy, labels.astype(np.int64, copy=False))

        end2end_ms = (time.time() - t_batch0) * 1000.0

        build["batch_starts"].append(batch_start)
        build["batch_raw"].append(raw_n)
        build["batch_kept_simd"].append(kept_simd)
        build["batch_simd_rm"].append(simd_rm)
        build["batch_oob_searched"].append(oob_searched)
        build["batch_oob_dist_dups"].append(oob_dist_dups)
        build["batch_oob_survivors_after_dist"].append(survivors_after_dist)
        build["batch_knownid_dups"].append(knownid_dups)
        build["batch_inserted"].append(inserted)
        build["batch_ms_minhash"].append(float(mh_ms))
        build["batch_ms_simd"].append(float(simd_ms))
        build["batch_ms_search"].append(float(search_ms))
        build["batch_ms_insert"].append(float(insert_ms))
        build["batch_ms_end2end"].append(float(end2end_ms))

        print(
            f"[BUILD] start={batch_start:<8d} raw={raw_n:<7d} kept_simd={kept_simd:<7d} simd_rm={simd_rm:<7d} "
            f"oob_searched={oob_searched:<7d} oob_dups={oob_dist_dups:<7d} knownid_dups={knownid_dups:<7d} inserted={inserted:<7d} "
            f"mh={mh_ms:.1f} simd={simd_ms:.1f} search={search_ms:.1f} ins={insert_ms:.1f} (ms)"
        )

    build_wall_ms = (time.time() - build_wall_t0) * 1000.0

    # Throughput (BUILD)
    build_total_raw = int(np.sum(build["batch_raw"])) if build["batch_raw"] else 0
    build_total_inserted = int(np.sum(build["batch_inserted"])) if build["batch_inserted"] else 0

    build_sumsteps_ms_with_search = float(
        np.sum(build["batch_ms_minhash"]) +
        np.sum(build["batch_ms_simd"]) +
        np.sum(build["batch_ms_search"]) +
        np.sum(build["batch_ms_insert"])
    )
    build_sumsteps_ms_no_search = float(
        np.sum(build["batch_ms_minhash"]) +
        np.sum(build["batch_ms_simd"]) +
        np.sum(build["batch_ms_insert"])
    )

    build["build_total_raw"] = build_total_raw
    build["build_total_inserted"] = build_total_inserted
    build["build_wall_ms"] = float(build_wall_ms)
    build["throughput_wall_rps_raw"] = (build_total_raw / (build_wall_ms / 1000.0)) if build_wall_ms > 0 else 0.0
    build["throughput_wall_rps_inserted"] = (build_total_inserted / (build_wall_ms / 1000.0)) if build_wall_ms > 0 else 0.0
    build["throughput_sumsteps_rps_raw_with_search"] = (build_total_raw / (build_sumsteps_ms_with_search / 1000.0)) if build_sumsteps_ms_with_search > 0 else 0.0
    build["throughput_sumsteps_rps_raw_no_search"] = (build_total_raw / (build_sumsteps_ms_no_search / 1000.0)) if build_sumsteps_ms_no_search > 0 else 0.0

    # --------------------------
    # PROBE phase (divergence)
    # --------------------------
    probe: Dict[str, Any] = {
        "phase": "probe",
        "batch_size": int(args.probe_batch),
        "ef_list": ef_list,
        "base_ef": int(base_ef),
        "probe_timing_ef": int(timing_ef),
        "dup_bits_threshold": int(args.dup_bits_threshold),
        "probe_efC": int(probe_efC),
        "batch_starts": [],
        "batch_raw": [],
        "batch_kept_simd": [],
        "batch_simd_rm": [],
        "batch_oob_dist_dups_base": [],
        "timing_search_ms_list":[],
        "batch_knownid_dups": [],
        "batch_inserted": [],
        "batch_ms_minhash": [],
        "batch_ms_simd": [],
        "batch_ms_search_timingef": [],
        "batch_ms_insert": [],
        "batch_ms_end2end": [],
    }

    # Accumulate probe search results (IDs) for ALL EFs, order-free comparison later
    # ef -> list of [n_batch, K] arrays, concatenated at end
    probe_ids_by_ef: Dict[int, List[np.ndarray]] = {ef: [] for ef in ef_list}
    probe_qids_all: List[np.ndarray] = []  # track which queries were compared (kept after SIMD)

    probe_wall_t0 = time.time()

    start_base = 0
    for df_pl in _rows_iter_pl(probe_ldf, probe_total, int(args.probe_batch)):
        batch_start = int(start_base)
        start_base += int(len(df_pl))

        df_pd = df_pl.to_pandas().sort_values("int_id_column").reset_index(drop=True)
        raw_n = int(len(df_pd))
        if raw_n == 0:
            continue

        t_batch0 = time.time()

        # MinHash
        t0 = time.time()
        items_tuples = parallel_minhash_fast(
            df_pd,
            perms,
            window_size=int(args.window),
            n_proc=int(threads),
            block_CH=8192,
            batch_size=int(args.mh_task_batch),
            verbose=False,
        )
        mh_ms = (time.time() - t0) * 1000.0

        items = to_items_tuples(items_tuples)
        items.sort(key=lambda d: d["int_id_column"])
        mh_mat = np.asarray([it["minhashes"] for it in items], dtype=np.uint64)

        # SIMD in-batch
        t0 = time.time()
        nested = simd.simd_fuzzy_deduplicationv7_unsorted_multi7A_V1(
            mh_mat,
            threshold=float(args.simd_threshold),
            num_bands=int(args.simd_bands),
            num_threads=int(threads),
        )
        simd_ms = (time.time() - t0) * 1000.0

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
        kept_simd = int(len(keep_idx))
        simd_rm = int(len(to_remove))

        # Record & continue if empty
        if kept_simd == 0:
            probe["batch_starts"].append(batch_start)
            probe["batch_raw"].append(raw_n)
            probe["batch_kept_simd"].append(0)
            probe["batch_simd_rm"].append(simd_rm)
            probe["batch_oob_dist_dups_base"].append(0)
            probe["batch_knownid_dups"].append(0)
            probe["batch_inserted"].append(0)
            probe["batch_ms_minhash"].append(float(mh_ms))
            probe["batch_ms_simd"].append(float(simd_ms))
            probe["batch_ms_search_timingef"].append(0.0)
            probe["batch_ms_insert"].append(0.0)
            probe["batch_ms_end2end"].append(float((time.time() - t_batch0) * 1000.0))
            continue

        kept_ids = np.array([int(items[i]["int_id_column"]) for i in keep_idx], dtype=np.int64)
        kept_mh  = mh_mat[keep_idx, :]
        XQ = minhashes_to_vectors(kept_mh, int(M_bits), int(mmh3_sd))

        # PROBE: run searches for ALL ef_list first (same index state), collect IDs for divergence
        # Then filter/insert survivors using BASE EF results only.
        # We map Ipos -> IDs using the CURRENT labels snapshot.
        labels_snapshot = labels  # do not modify until after all searches

        # We also collect "timing ef" ms as the probe timing search component
        timing_search_ms = 0.0
        timing_search_ms_list=[]
        D_base: Optional[np.ndarray] = None
        Ipos_base: Optional[np.ndarray] = None

        for ef in ef_list:
            index.hnsw.efConstruction = int(probe_efC)
            index.hnsw.efSearch = int(ef)

            # New: top-level fields (no-op for IndexBinaryHNSW but kept)
            index.efConstruction = int(probe_efC)
            index.efSearch = ef


            # Warmup search: trigger pb warm-build with minimal workload
            t0 = time.time()
            if XQ.shape[0] >= 1:
                _D_warm, _I_warm = index.search(XQ[:2], 1)
            dt_ms = (time.time() - t0) * 1000.0
            print("==========warm dt_ms==========")
            print(dt_ms)

             # Debug: show existing efSearch/efConstruction
            print("==========warm dt_ms===PROBE SEARCH=======")
            print("efSearch:", index.hnsw.efSearch)
            print("efConstruction:", index.hnsw.efConstruction)



            t0 = time.time()
            D, Ipos = index.search(XQ, int(args.topk))
            dt_ms = (time.time() - t0) * 1000.0

            timing_search_ms_list.append(float(dt_ms))

            # Map positions -> IDs (this is what you compare, order-free)
            I_ids = np.where(Ipos >= 0, labels_snapshot[Ipos], -1).astype(np.int64, copy=False)
            probe_ids_by_ef[int(ef)].append(I_ids)

            if int(ef) == int(timing_ef):
                timing_search_ms = float(dt_ms)

            if int(ef) == int(base_ef):
                D_base = D
                Ipos_base = Ipos
                print(D)
                print(Ipos)

        probe_qids_all.append(kept_ids.copy())

        # BASE EF duplicate filter + insert survivors (NO re-search)
        assert D_base is not None and Ipos_base is not None

        if D_base.size:
            min_d = D_base.min(axis=1)
            dup_mask_dist = (min_d <= int(args.dup_bits_threshold))
        else:
            dup_mask_dist = np.zeros((XQ.shape[0],), dtype=bool)

        oob_dist_dups_base = int(np.sum(dup_mask_dist))
        survivors_after_dist = int(np.sum(~dup_mask_dist))

     

        inserted = 0
        knownid_dups = 0
        insert_ms = 0.0

        if survivors_after_dist > 0:
            ids_cand = kept_ids[~dup_mask_dist]
            Xcand = XQ[~dup_mask_dist, :]

            new_mask = np.array([int(iid) not in known_ids for iid in ids_cand], dtype=bool)
            knownid_dups = int(np.sum(~new_mask))

            if new_mask.any():
                XInsert = Xcand[new_mask, :]
                ids_to_insert = ids_cand[new_mask]

                
                index.hnsw.efConstruction = int(probe_efC)
                index.hnsw.efSearch = 10

                # New: top-level fields (no-op for IndexBinaryHNSW but kept)
                index.efConstruction = int(probe_efC)
                index.efSearch =10

                # Debug: show existing efSearch/efConstruction
                     # Debug: show existing efSearch/efConstruction
                print("==========warm dt_ms===PROBE INSERT=======")
                print("efSearch:", index.hnsw.efSearch)
                print("efConstruction:", index.hnsw.efConstruction)



                t0 = time.time()
                index.add(XInsert)
                insert_ms = (time.time() - t0) * 1000.0

                labels = np.concatenate([labels, ids_to_insert.astype(np.int64, copy=False)])
                known_ids.update(map(int, ids_to_insert.tolist()))
                inserted = int(XInsert.shape[0])

                faiss.write_index_binary(index, str(args.index_path))
                np.save(args.labels_npy, labels.astype(np.int64, copy=False))

        end2end_ms = (time.time() - t_batch0) * 1000.0

        #  n_duplicates_total += n_dup_batch

        probe["batch_starts"].append(batch_start)
        probe["batch_raw"].append(raw_n)
        probe["batch_kept_simd"].append(kept_simd)
        probe["batch_simd_rm"].append(simd_rm)
        probe["batch_oob_dist_dups_base"].append(oob_dist_dups_base)
        probe["timing_search_ms_list"].append(np.array(timing_search_ms_list).tolist())

        probe["batch_knownid_dups"].append(knownid_dups)
        probe["batch_inserted"].append(inserted)
        probe["batch_ms_minhash"].append(float(mh_ms))
        probe["batch_ms_simd"].append(float(simd_ms))
        probe["batch_ms_search_timingef"].append(float(timing_search_ms))
        probe["batch_ms_insert"].append(float(insert_ms))
        probe["batch_ms_end2end"].append(float(end2end_ms))

        print(
            f"[PROBE] start={batch_start:<8d} raw={raw_n:<7d} kept_simd={kept_simd:<7d} simd_rm={simd_rm:<7d} "
            f"base_oob_dups={oob_dist_dups_base:<7d} knownid_dups={knownid_dups:<7d} inserted={inserted:<7d} "
            f"mh={mh_ms:.1f} simd={simd_ms:.1f} search({timing_ef})={timing_search_ms:.1f} ins={insert_ms:.1f} (ms)"
        )

    probe_wall_ms = (time.time() - probe_wall_t0) * 1000.0

    # Throughput (PROBE)
    probe_total_raw = int(np.sum(probe["batch_raw"])) if probe["batch_raw"] else 0
    probe_total_kept = int(np.sum(probe["batch_kept_simd"])) if probe["batch_kept_simd"] else 0

    probe_sumsteps_ms = float(
        np.sum(probe["batch_ms_minhash"]) +
        np.sum(probe["batch_ms_simd"]) +
        np.sum(probe["batch_ms_search_timingef"]) +
        np.sum(probe["batch_ms_insert"])
    )

    probe["probe_total_raw"] = probe_total_raw
    probe["probe_total_kept_simd"] = probe_total_kept
    probe["probe_wall_ms"] = float(probe_wall_ms)
    probe["throughput_wall_rps_raw"] = (probe_total_raw / (probe_wall_ms / 1000.0)) if probe_wall_ms > 0 else 0.0
    probe["throughput_wall_rps_kept_simd"] = (probe_total_kept / (probe_wall_ms / 1000.0)) if probe_wall_ms > 0 else 0.0
    probe["throughput_sumsteps_rps_raw_timingef"] = (probe_total_raw / (probe_sumsteps_ms / 1000.0)) if probe_sumsteps_ms > 0 else 0.0

    # --------------------------
    # Divergence computation (order-free, whole probe)
    # --------------------------
    # Concatenate per-ef arrays
    probe_ids_concat: Dict[int, np.ndarray] = {}
    for ef in ef_list:
        if probe_ids_by_ef[ef]:
            probe_ids_concat[ef] = np.vstack(probe_ids_by_ef[ef])
        else:
            probe_ids_concat[ef] = np.zeros((0, int(args.topk)), dtype=np.int64)

    # Make sure shapes align
    N_comp = probe_ids_concat[ef_list[0]].shape[0]
    for ef in ef_list[1:]:
        if probe_ids_concat[ef].shape[0] != N_comp:
            raise RuntimeError(f"Probe accumulation mismatch: ef{ef} has {probe_ids_concat[ef].shape[0]} queries, expected {N_comp}")

    divergence = divergence_order_free_summary(probe_ids_concat, topk=int(args.topk))

    # Optional: save raw probe ID matrices for later verification
    if args.save_probe_ids_npz:
        out_npz = outdir / "metrics" / "probe_divergence" / f"{args.series_tag}_probe_ids_topk{int(args.topk)}.npz"
        np_save = {f"ef{ef}": probe_ids_concat[ef] for ef in ef_list}
        np.savez_compressed(out_npz, **np_save)
        divergence["saved_probe_ids_npz"] = str(out_npz)

    # --------------------------
    # Write metrics
    # --------------------------
    out_metrics = outdir / "metrics" / "probe_divergence" / f"{args.series_tag}.json"
    payload = {
        "series": str(args.series_tag),
        "IDX": (int(args.IDX) if args.IDX is not None else None),
        "index_path": str(args.index_path),
        "labels_path": str(args.labels_npy),
        "corpus_parquet": str(args.corpus_parquet),
        "probe_parquet": str(args.probe_parquet),
        "M": int(args.M),
        "topk": int(args.topk),
        "ef_list": ef_list,
        "base_ef": int(base_ef),
        "probe_timing_ef": int(timing_ef),
        "build": build,
        "probe": probe,
        "divergence_order_free": divergence,
    }
    write_json(out_metrics, payload)
    print(f"✓ Wrote metrics → {out_metrics}")


if __name__ == "__main__":
    main()
