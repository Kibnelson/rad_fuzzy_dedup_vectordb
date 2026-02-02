#!/usr/bin/env python3
# Batched MINHASH_LSH query + SIMD in-batch dedup + conditional insert (Milvus).
from __future__ import annotations
import argparse, os, re, json, time, math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time, datetime as dt
import numpy as np
import polars as pl
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import mmh3
from pymilvus import MilvusClient

# SIMD dedup (compiled module)
import simd_fuzzy_deduplicationv7_unsorted_multi7A_V1 as simd
from pathlib import Path
from typing import List
import polars as pl


def append_insert_docs(
    qparq: Path,
    mainparq: Path,
    insert_ids: List[int],
    id_col: str = "int_id_column",
    text_col: str = "contents",
    dedup: bool = True,
) -> None:
    # 1. Read the source parquet with all possible docs
    df_src = pl.read_parquet(qparq)

    # 2. Filter rows whose id is in insert_ids
    df_new = (
        df_src
        .filter(pl.col(id_col).is_in(insert_ids))
        [[id_col, text_col]]  # keep only these columns
    )

    # 3. If mainparq already exists, append; otherwise start from df_new
    if mainparq.exists():
        df_main = pl.read_parquet(mainparq)
        df_out = pl.concat([df_main, df_new], how="vertical_relaxed")
    else:
        df_out = df_new

    # 4. Optional: deduplicate by id so we don't keep multiple copies
    if dedup:
        df_out = df_out.unique(subset=[id_col], keep="last")

    # 5. Write back to the same path for next usage
    df_out.write_parquet(mainparq)

# ---------- FS helpers ----------
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
    perms = np.load(outdir / "manifests" / "permutations.npy").astype(np.uint64)
    return spec, perms

# ---------- MinHash helpers ----------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize_words(s: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(s or "")]

def shingles(words: List[str], k: int) -> List[str]:
    if not words:
        return []
    if len(words) < k:
        return [" ".join(words)]
    return [" ".join(words[i:i+k]) for i in range(len(words)-k+1)]

def minhash_signature(text: str, perms: np.ndarray, win: int) -> np.ndarray:
    """
    MinHash signature aligned with build_index_simple.py / Milvus ingest:
    - split on whitespace
    - word shingles of size `win`
    - mmh3.hash(s, signed=False) → uint64
    - (hv * perms) >> 32, take per-lane min
    - returns uint32[K]
    """
    import numpy as _np

    words = (text or "").split()
    n = len(words)
    K = perms.shape[0]

    if n == 0:
        return _np.zeros(K, dtype=_np.uint32)

    shingles = [
        " ".join(words[i:i+win])
        for i in range(max(1, n - win + 1))
    ]

    hv = _np.array(
        [mmh3.hash(s, signed=False) for s in shingles],
        dtype=_np.uint64
    )
    a = perms.astype(_np.uint64)

    best = _np.full((K,), _np.uint64((1 << 64) - 1), dtype=_np.uint64)
    CH = 8192
    for start in range(0, hv.shape[0], CH):
        sl = hv[start:start+CH][:, None]          # (batch, 1)
        vals = (sl * a[None, :]) >> _np.uint64(32)  # (batch, K)
        best = _np.minimum(best, vals.min(axis=0))

    return best.astype(_np.uint32)

def mh_words_to_bytes(mh: np.ndarray, bit_width: int) -> bytes:
    if bit_width == 64:
        return mh.astype(">u8").tobytes()
    if bit_width == 32:
        return mh.astype(">u4").tobytes()
    if bit_width == 16:
        return mh.astype(">u2").tobytes()
    if bit_width == 8:
        return mh.astype(np.uint8).tobytes()
    raise ValueError("mh_bit_width must be one of {8,16,32,64}")

def mh_words_to_token_str(mh: np.ndarray) -> str:
    return " ".join(str(int(x)) for x in mh.tolist())

# ---------- Parallel MinHash (range-based, shared globals) ----------
_G_PERMS: np.ndarray | None = None
_G_WINDOW: int | None = None
_G_QTEXTS: np.ndarray | None = None
_G_QIDS: np.ndarray | None = None

def _init_worker_mh(perms: np.ndarray,
                    window_size: int,
                    qtexts: np.ndarray,
                    qids: np.ndarray) -> None:
    """
    Initializer for worker processes:
    broadcasts permutations + query texts + ids once per process.
    """
    global _G_PERMS, _G_WINDOW, _G_QTEXTS, _G_QIDS
    _G_PERMS = perms
    _G_WINDOW = int(window_size)
    _G_QTEXTS = qtexts
    _G_QIDS = qids

def _minhash_worker_range(start: int, end: int) -> List[Dict[str, Any]]:
    """
    Worker: compute MinHash for rows [start:end) from global arrays.
    Returns list of dicts: {"minhashes": mh, "qid": ..., "qtext": ...}
    """
    assert _G_PERMS is not None
    assert _G_WINDOW is not None
    assert _G_QTEXTS is not None
    assert _G_QIDS is not None

    perms = _G_PERMS
    win = _G_WINDOW
    qtexts = _G_QTEXTS
    qids = _G_QIDS

    out: List[Dict[str, Any]] = []
    for idx in range(start, end):
        content = qtexts[idx]
        qid = int(qids[idx])
        mh = minhash_signature(content, perms, win=win)
        out.append({"minhashes": mh, "qid": qid, "qtext": content})
    return out

def parallel_minhash(
    df_pd: pd.DataFrame,
    permutations: np.ndarray,
    *,
    window_size: int,
    batch_size: int,
    n_proc: int,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Parallel MinHash over a query batch, using range-based tasks and
    shared globals.

    - df_pd must have columns: "qid", "qtext"
    - rows are evenly split across `n_proc` workers
    """
    N = len(df_pd)
    if N == 0:
        return []

    qtexts = df_pd["qtext"].values
    qids = df_pd["qid"].values

    # Decide number of workers based on n_proc
    workers = max(1, int(n_proc) if n_proc is not None else 1)
    workers = min(workers, N)

        # Build index ranges for tasks (avoid pickling large arrays)
    bs = max(1, batch_size)
    ranges = [(off, min(off + bs, N)) for off in range(0, N, bs)]
    num_batches = len(ranges)





    results: List[Dict[str, Any]] = []

    if workers == 1 or num_batches == 1:
        _init_worker_mh(permutations, window_size, qtexts, qids)
        for s, e in ranges:
            results.extend(_minhash_worker_range(s, e))
        return results

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker_mh,
        initargs=(permutations, int(window_size), qtexts, qids),
    ) as pool:
        fut_meta: Dict[Any, Tuple[int, int, int, float]] = {}
        for i, (s, e) in enumerate(ranges, 1):
            t_submit = time.perf_counter()
            fut = pool.submit(_minhash_worker_range, s, e)
            fut_meta[fut] = (i, s, e, t_submit)
        done = 0
        for fut in as_completed(fut_meta):
            i, s, e, t_submit = fut_meta[fut]
            try:
                chunk = fut.result()
            except Exception as ex:
                raise RuntimeError(
                    f"_minhash_worker_range failed for [{s}:{e})"
                ) from ex
            results.extend(chunk)
            done += 1
            if verbose:
                print(
                    f"[parallel_minhash] done {done}/{num_batches} "
                    f"range=[{s}:{e}) "
                    f"wait={(time.perf_counter()-t_submit):.3f}s"
                )
    total_s = time.perf_counter() - t0
    if verbose:
        print(
            f"[parallel_minhash] total={total_s:.3f}s "
            f"({total_s/60:.2f} min), items={len(results)}"
        )
    return results

# ---------- Polars streaming ----------
def _rows_iter_pl(ldf: pl.LazyFrame, total: int, batch_rows: int):
    start = 0
    while start < total:
        stop = min(start + batch_rows, total)
        yield ldf.slice(start, stop - start).collect(engine="streaming")
        start = stop

# ---------- Result normalization ----------
def _hit_get_id(hit, pk_field):
    if isinstance(hit, dict):
        ent = hit.get("entity") or {}
        return int(ent.get(pk_field, hit.get("id")))
    try:
        return int(getattr(hit, "entity", {}).get(pk_field, getattr(hit, "id")))
    except Exception:
        return int(hit.id)

def _hit_get_distance(hit):
    if isinstance(hit, dict):
        return float(hit.get("distance", float("nan")))
    try:
        return float(getattr(hit, "distance"))
    except Exception:
        return float("nan")

def normalize_hits(res, pk_field: str):
    out = []
    for hits in res:
        row = []
        for h in hits:
            row.append({"id": _hit_get_id(h, pk_field), "distance": int(h.get("distance")) })
        out.append(row)
    return out

# ---------- Benchmark helpers ----------
def load_ground_truth(gt_path: Optional[Path]) -> Optional[Dict[int, List[int]]]:
    if not gt_path:
        return None
    raw = read_json(gt_path)
    return {int(k): list(map(int, v)) for k, v in raw.items()}

from collections import defaultdict

def compute_metric(
    I: List[List[int]],
    q_ids: List[int],
    gt: Dict[int, List[int]],
    k: int
) -> Tuple[Optional[float], str, int, Optional[int]]:
    """
    Global (total) metric with bi-directional GT.
    """
    if not gt:
        return None, "none", 0, None

    # Build reverse mapping: value -> [keys]
    rev = defaultdict(list)
    for kf, vs in gt.items():
        for v in vs:
            rev[int(v)].append(int(kf))

    # Resolve targets per query (prefer forward; fallback to reverse)
    targets_per_q: List[List[int]] = []
    for qid in q_ids:
        t = gt.get(qid)
        if not t:
            t = rev.get(qid, [])
        targets_per_q.append([int(x) for x in t])

    # Decide mode (single-label if all non-empty target lists have length ≤ 1)
    non_empty = [t for t in targets_per_q if t]
    single = all(len(t) <= 1 for t in non_empty)

    # Expected total from GT (restricted to queries we evaluated)
    if single:
        expected_total = len(non_empty)  # 1 expected per non-empty query
    else:
        expected_total = sum(len(t) for t in non_empty)  # count all targets

    if expected_total == 0:
        return None, ("hit_total" if single else "recall_total"), 0, 0

    # Count correct found
    correct_found = 0
    for qi, targets in enumerate(targets_per_q):
        if not targets:
            continue
        candidates = set(I[qi][:k]) if qi < len(I) else set()
        if single:
            # any one match counts as 1
            true_id = targets[0]
            if true_id in candidates:
                correct_found += 1
        else:
            # count all matches (intersection)
            correct_found += len(candidates & set(targets))

    score = correct_found / float(expected_total)
    return score, ("hit_total" if single else "recall_total"), expected_total, correct_found

def weighted_mean_ci(x, w, z=1.96):
    x = np.asarray(x, float); w = np.asarray(w, float)
    W = w.sum()
    mu = (w * x).sum() / W if W > 0 else 0.0
    denom = W - (w @ w) / W if W > 0 else 0.0
    if denom <= 0:
        return mu, mu, mu
    s2 = (w * (x - mu)**2).sum() / denom
    n_eff = (W**2) / (w @ w)
    se = math.sqrt(s2 / n_eff) if n_eff > 0 else 0.0
    return mu, mu - z*se, mu + z*se

def sort_and_extrema(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return x, {"min": 0.0, "max": 0.0, "min_idx": -1, "max_idx": -1}
    order = np.argsort(x)
    xs = x[order]
    return xs, {"min": float(xs[0]), "max": float(xs[-1]), "min_idx": int(order[0]), "max_idx": int(order[-1])}

def parquet_num_rows(p: Path) -> int:
    """
    Return total rows in a Parquet file using metadata only.
    Falls back to Polars if PyArrow isn't available.
    """
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(str(p)).metadata.num_rows
    except Exception:
        import polars as pl  # type: ignore
        return pl.scan_parquet(str(p)).select(pl.len()).collect(streaming=True).item()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser("Milvus MINHASH_LSH — batched query + SIMD dedup + conditional insert")
    ap.add_argument("--outdir", required=True, type=Path)

    # Milvus
    ap.add_argument("--uri", default=os.environ.get("MILVUS_URI", "http://127.0.0.1:19530"))
    ap.add_argument("--collection", default="corpus_minhash_lsh")
    ap.add_argument("--pk_field", default="doc_id")
    ap.add_argument("--partition", type=str, default=None)

    # Queries
    ap.add_argument("--queries_parquet", required=True)
    ap.add_argument("--corpus_parquet", type=Path,
                    help="Parquet with all indexed docs (for refine_k sizing)")
    ap.add_argument("--refine_k_fraction", type=float, required=True, default=None,
                    help="If set, refine_k = clamp(ceil(fraction * corpus_rows), topk..refine_k_cap)")

    ap.add_argument("--id_col", default="int_id_column")
    ap.add_argument("--text_col", default="contents")
    ap.add_argument("--query_limit", type=int, default=0, help="0 = all")

    # MinHash params
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--mh_bit_width", type=int, default=32, choices=[8,16,32,64])

    # Parallel MinHash + batching
    ap.add_argument("--mh_batch", type=int, default=10_000,
                    help="(deprecated, unused) kept for backward compatibility")
    ap.add_argument("--query_batch", type=int, default=10_000,
                    help="Queries per Polars streaming batch")

    # Unified thread knob for MinHash + SIMD
    ap.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 8)),
        help="Number of worker processes / SIMD threads to use",
    )

    # SIMD in-batch dedup
    ap.add_argument("--simd_threshold", type=float, default=0.7)
    ap.add_argument("--simd_bands", type=int, default=14)
    ap.add_argument("--save_dedup_drops", action="store_true")

    # Search
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--refine_k", type=int, default=100, help=">0 enables refined Jaccard")

    # INSERT control
    ap.add_argument("--distance_threshold", type=float, default=0.7,
                    help="Drop if top-1 MHJACCARD distance ≤ threshold; otherwise INSERT")
    ap.add_argument("--insert_batch", type=int, default=10_000, help="Rows per Milvus insert()")

    # Ground truth + outputs
    ap.add_argument("--gt_json", type=Path, default=None, help="qid -> [true ids]")
    ap.add_argument("--series_tag", default="milvus_minhash_lsh_query_insert_simd")
    ap.add_argument("--out_metrics", type=Path, default=None)
    ap.add_argument("--save_neighbors", action="store_true")
    ap.add_argument("--neighbors_dir", type=Path, default=None)

    args = ap.parse_args()

    outdir = args.outdir.resolve()
    ensure_dir(outdir / "metrics" / "recall_curves")
    if args.save_dedup_drops:
        ensure_dir(outdir / "results" / "dedup_drops" / args.series_tag)

    # Load spec & perms
    spec, perms = load_spec_and_perms(outdir)
    K = perms.shape[0]
    print(">>>>>>>>")
    print(K)
    dim_bits = 112 * int(args.mh_bit_width)
    expected_bytes = dim_bits // 8
    print(f"Expect bytes/vector={expected_bytes} (K={K}, bw={args.mh_bit_width})")
    print(f"Using threads={args.threads} for MinHash + SIMD")

    # Milvus
    client = MilvusClient(uri=args.uri)
    client.load_collection(args.collection)
    print(f"Connected → {args.uri}; collection loaded: {args.collection}")

    # Queries stream
    qparq = Path(args.queries_parquet)
    if not qparq.exists():
        raise FileNotFoundError(f"Queries parquet not found: {qparq}")
    q_ldf = (
        pl.scan_parquet(qparq)
          .select(pl.col(args.id_col).alias("qid"), pl.col(args.text_col).alias("qtext"))
          .with_columns(pl.col("qid").cast(pl.Int64), pl.col("qtext").cast(pl.Utf8))
    )
    if args.query_limit and args.query_limit > 0:
        q_ldf = q_ldf.limit(args.query_limit)
    q_total = q_ldf.select(pl.len()).collect(engine="streaming").item()
    print(f"Planned queries: {q_total:,} from {qparq}")


    RANDOM_COL = "_rand"

  





    # Decide refine_k
    if args.refine_k_fraction is not None:
        corpus_parquet = args.corpus_parquet or (outdir / "cache" / "corpus.parquet")
        if not corpus_parquet.exists():
            raise FileNotFoundError(f"Corpus parquet not found for refine_k sizing: {corpus_parquet}")
        total_rows = parquet_num_rows(corpus_parquet)
        refine_k_dynamic = int(math.ceil(total_rows * float(args.refine_k_fraction)))
        print(
            f"• Dynamic refine_k: rows={total_rows:,}, "
            f"fraction={args.refine_k_fraction} → refine_k={refine_k_dynamic} "
            f"(clamped [{args.topk}])"
        )
    else:
        refine_k_dynamic = int(args.refine_k)
        print(f"• Static refine_k: {refine_k_dynamic}")

    # Outputs
    out_metrics = args.out_metrics or (outdir / "metrics" / "recall_curves" / f"{args.series_tag}.json")
    neighbors_dir = args.neighbors_dir or (outdir / "results" / "neighbors" / args.series_tag)
    if args.save_neighbors:
        ensure_dir(neighbors_dir)

    # Search params (stable)
    sp = {"metric_type": "MHJACCARD", "params": {"mh_lsh_batch_search": True}}
    # if refine_k_dynamic and refine_k_dynamic > 0:
    # sp = {
    #     "metric_type": "MHJACCARD",
    #     "params": {"mh_search_with_jaccard": False, "mh_lsh_batch_search": True},
    # }

    # Benchmark accumulators
    gt = load_ground_truth(args.gt_json)
    per_batch_avg_ms: List[float] = []
    per_batch_dt_ms: List[float] = []
    per_insert_ms: List[float] = []
    per_batch_sizes: List[int] = []
    per_batch_total_s: List[float] = []
    per_batch_dedup_ms: List[float] = []
    per_batch_dedup_size: List[int] = []
    per_minhash_ms: List[float] = []

    all_qids: List[int] = []
    all_neighbors: List[List[int]] = []

    insert_ids: List[int] = []


    to_insert_rows: List[Dict[str, Any]] = []
    n_inserted_total = 0
    n_dropped_by_dist = 0

    start_idx = 0
    batch_no = 0

    t_all0 = time.time()

    for qdf in _rows_iter_pl(q_ldf, q_total, args.query_batch):
        batch_no += 1
        df_pd = qdf.rename({"qid": "qid", "qtext": "qtext"}).to_pandas()
        df_pd = df_pd.sort_values("qid").reset_index(drop=True)

        # 1) parallel MinHash
        t_mh0 = time.time()
        items = parallel_minhash(
            df_pd,
            permutations=perms,
            window_size=args.window,
            batch_size=args.mh_batch,
            n_proc=args.threads,
            verbose=False,
        )
        mh_ms = (time.time() - t_mh0) * 1000.0
        per_minhash_ms.append(mh_ms)

        if not items:
            start_idx += len(df_pd)
            continue

        # 2) SIMD in-batch dedup
        items.sort(key=lambda d: d["qid"])
        mh_mat = np.asarray([it["minhashes"] for it in items], dtype=np.uint64)
        t_batch0 = time.time()
        nested = simd.simd_fuzzy_deduplicationv7_unsorted_multi7A_V1(
            mh_mat,
            threshold=float(args.simd_threshold),
            num_bands=int(args.simd_bands),
            num_threads=int(args.threads),
        )

        dedup_dt_ms = (time.time() - t_batch0) * 1000.0
        per_batch_dedup_ms.append(dedup_dt_ms)

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

        kept_ids = [int(items[i]["qid"]) for i in keep_idx]
        kept_mh = mh_mat[keep_idx, :]
        all_qids.extend(kept_ids)

        # Prepare Milvus binaries for search and potential insert
        queries = []
        for i in range(kept_mh.shape[0]):
            mh = kept_mh[i, :].astype(np.uint32)
            qsig = mh_words_to_bytes(mh, args.mh_bit_width)
            if len(qsig) != expected_bytes:
                raise RuntimeError(f"encoded bytes {len(qsig)} != expected {expected_bytes}")
            queries.append(qsig)

        queries_augmented = list(queries)
        # print(queries_augmented)

        n_orig = len(queries)
        print(n_orig)

        # 3) Milvus search  limit=args.topk,
        t0 = time.time()
        res = client.search(
            collection_name=args.collection,
            data=queries_augmented,
            anns_field="minhash_signature",
            search_params=sp,
            limit=args.topk,
            output_fields=[args.pk_field],
            consistency_level="Bounded",
        )


        dt_ms = (time.time() - t0) * 1000.0
        per_batch_dt_ms.append(dt_ms)

        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[CLIENT] BATCH {batch_no} finished at {now} search_total_ms={dt_ms:.2f}")


        

        # Normalize + collect neighbors
        full = normalize_hits(res, args.pk_field)
        I_ids = [[int(h["id"]) for h in hits] for hits in full]
        all_neighbors.extend(I_ids)


        # 4) Conditional INSERT
        for idx_local, qid in enumerate(kept_ids):
            hits = full[idx_local]
            drop_for_similarity = False

            if hits:
                top1_dist = float(hits[0]["distance"])
                if not np.isnan(top1_dist) and top1_dist >= float(args.distance_threshold):
                    drop_for_similarity = True

            if drop_for_similarity:
                n_dropped_by_dist += 1
                continue  # do not insert

            mh = kept_mh[idx_local, :].astype(np.uint32)
            row = {
                "doc_id": int(qid),
                "minhash_signature": mh_words_to_bytes(mh, args.mh_bit_width),
                "token_set": mh_words_to_token_str(mh),
            }
            insert_ids.append(int(qid))
            to_insert_rows.append(row)

            if len(to_insert_rows) >= args.insert_batch:
                try:
                    t2 = time.time()
                    client.insert(
                        collection_name=args.collection,
                        data=to_insert_rows,
                        partition_name=(args.partition if args.partition else None),
                    )
                    insert_ms = (time.time() - t2) * 1000.0
                    per_insert_ms.append(insert_ms)

                    n_inserted_total += len(to_insert_rows)
                    to_insert_rows.clear()
                except Exception as e:
                    print(f" insert batch failed: {e}")
                    to_insert_rows.clear()

        per_batch_sizes.append(len(queries))
        per_batch_avg_ms.append(dt_ms / max(1, len(queries)))
        per_batch_total_s.append(time.time() - t_batch0)

        print(
            f"[batch {batch_no}] kept={len(queries)} (dedup dropped {len(items)-len(queries)}) "
            f"MH={mh_ms:.1f}ms dedup={dedup_dt_ms:.1f}ms search={dt_ms:.1f}ms "
            f"to_insert_buffer={len(to_insert_rows)} dropped_by_dist_total={n_dropped_by_dist}"
        )

        start_idx += len(df_pd)

    # Final flush of any remaining rows
    if to_insert_rows:
        try:
            t2 = time.time()
            client.insert(
                collection_name=args.collection,
                data=to_insert_rows,
                partition_name=(args.partition if args.partition else None),
            )
            insert_ms = (time.time() - t2) * 1000.0
            per_insert_ms.append(insert_ms)

            n_inserted_total += len(to_insert_rows)
            to_insert_rows.clear()
        except Exception as e:
            print(f" final insert batch failed: {e}")

    # ----- Metrics -----
    total_runtime_s = time.time() - t_all0
    minhahash_total = float(np.sum(per_minhash_ms)) if per_minhash_ms else 0.0
    dedup_total_ms = float(np.sum(per_batch_dedup_ms)) if per_batch_dedup_ms else 0.0
    total_dt_ms = float(np.sum(per_batch_dt_ms)) if per_batch_dt_ms else 0.0
    total_insert_ms = float(np.sum(per_insert_ms)) if per_insert_ms else 0.0
    n_total = int(np.sum(per_batch_sizes)) if per_batch_sizes else 0

    #  get

    qparq = Path(args.queries_parquet)
    mainparq = Path(args.queries_parquet)
    append_insert_docs(qparq, mainparq, insert_ids)
    


    # Accuracy
    score: Optional[float] = None
    mode = "none"
    counted = 0
    successes: Optional[int] = None
    if gt is not None:
        score, mode, counted, successes = compute_metric(all_neighbors, all_qids, gt, args.topk)

    results = {
        "series": args.series_tag,
        "index": "MINHASH_LSH",
        "metric": "MHJACCARD",
        "topk": args.topk,
        "refine_k": args.refine_k,
        "k": int(K),
        "mh_bit_width": int(args.mh_bit_width),
        "window": int(args.window),
        "summary": {
            "total_batches": len(per_batch_sizes),
            "total_queries": n_total,
            "search_total_ms": total_dt_ms,
            "total_insert_ms": total_insert_ms,
            "minhahash_total": minhahash_total,
            "throughput_all_steps": (
                int(np.sum(per_batch_sizes)) / (total_runtime_s / 1000.0)
            ) if total_runtime_s > 0 else 0.0,
            "dedup_total_ms": dedup_total_ms,
            "n_inserted_total": int(n_inserted_total),
            "n_dropped_by_distance": int(n_dropped_by_dist),
            "distance_threshold": float(args.distance_threshold),
            "total_runtime_s": float(total_runtime_s),
            "insert_batch": int(args.insert_batch),
        },
        "accuracy": {
            "mode": mode,
            "score": score,
            "n_counted": counted,
            "successes": (int(successes) if successes is not None else None),
        },
    }

    print("\n===== SUMMARY =====")
    print(json.dumps(results["summary"], indent=2))
    print("\n===== ACCURACY =====")
    print(json.dumps(results["accuracy"], indent=2))

    write_json(out_metrics, results)
    print(f"✓ Wrote metrics → {out_metrics}")

if __name__ == "__main__":
    main()
