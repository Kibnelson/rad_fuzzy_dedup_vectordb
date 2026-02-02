#!/usr/bin/env python3
# Batched MINHASH_LSH ingest from a parquet (id_col, text_col),
# using spec + permutations from outdir/manifests (like build_index_simple.py).
from __future__ import annotations
import argparse, os, re, json, time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import polars as pl
import mmh3
from pymilvus import MilvusClient, DataType
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------- FS helpers ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_spec_and_perms(outdir: Path):
    spec_path = outdir / "manifests" / "sketch_spec.json"
    perms_path = outdir / "manifests" / "permutations.npy"
    if not spec_path.exists() or not perms_path.exists():
        raise SystemExit(
            f"Missing manifests in {outdir}.\n"
            f"  expected: {spec_path}\n"
            f"            {perms_path}\n"
            f"Create them first (same K/seed you’ll use for query)."
        )
    spec  = read_json(spec_path)
    perms = np.load(perms_path).astype(np.uint64)
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
    return [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]


def minhash_signature(text: str, perms: np.ndarray, win: int) -> np.ndarray:
    """
    MinHash signature that matches _generate_word_shingles_old_single1
    in the FAISS build script:
      - word-level shingles via `text.split()`
      - window size = win
      - mmh3.hash(s, signed=False) -> uint64
      - (hv * perms) >> 32, then column-wise min
    """
    import numpy as _np

    # 1) Tokenize like in the FAISS script
    words = (text or "").split()
    n = len(words)
    K = perms.shape[0]

    if n == 0:
        return _np.zeros(K, dtype=_np.uint32)

    # 2) Build k-shingles with the same window semantics
    shingles = [
        " ".join(words[i:i + win])
        for i in range(max(1, n - win + 1))
    ]

    # 3) Hash shingles with mmh3.hash(s, signed=False) → uint64
    hv = _np.array(
        [mmh3.hash(s, signed=False) for s in shingles],
        dtype=_np.uint64,
    )

    # 4) Same multiply-and-shift trick as _generate_word_shingles_old_single1
    a = perms.astype(_np.uint64)
    best = _np.full((K,), _np.uint64((1 << 64) - 1), dtype=_np.uint64)

    CH = 8192
    for start in range(0, hv.shape[0], CH):
        sl = hv[start:start + CH][:, None]      # (chunk_size, 1)
        vals = (sl * a[None, :]) >> _np.uint64(32)  # (chunk_size, K)
        best = _np.minimum(best, vals.min(axis=0))

    return best.astype(_np.uint32)


def minhash_signature1(text: str, perms: np.ndarray, win: int) -> np.ndarray:
    # Unused helper kept for parity with older versions; not called.
    import numpy as _np
    words = text.split()
    n = len(words)
    K = perms.shape[0]
    if n == 0:
        return [0] * K
    shingles = [delimiter.join(words[i:i+window_size]) for i in range(max(1, n - window_size + 1))]
    hv = _np.array([mmh3.hash(s, signed=False) for s in shingles], dtype=_np.uint64)
    a  = permutations.astype(_np.uint64)
    best = _np.full((K,), _np.uint64((1 << 64) - 1), dtype=_np.uint64)
    CH = 8192
    for start in range(0, hv.shape[0], CH):
        sl = hv[start:start+CH][:, None]
        vals = (sl * a[None, :]) >> _np.uint64(32)
        best = _np.minimum(best, vals.min(axis=0))
    return best.astype(_np.uint32).tolist()

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


def ensure_minhash_index(client, collection_name: str, field_name: str,
                         mh_bit_width: int, bands: int):
    # Does an index already exist?
    try:
        idxes = client.list_indexes(collection_name)
    except Exception:
        idxes = []
    # (Index creation is done in main() below; this helper is kept for parity.)

# =========================
# NEW: parallel MinHash (same pattern as FAISS build/query)
# =========================

_G_PERMS: np.ndarray | None = None
_G_WINDOW: int | None = None
_G_TEXTS: np.ndarray | None = None   # np.ndarray[object] of strings
_G_IDS:   np.ndarray | None = None   # np.ndarray[int]


def _init_worker_mh(perms: np.ndarray,
                    window_size: int,
                    texts: np.ndarray,
                    ids: np.ndarray):
    """
    Initializer for worker processes:
    broadcasts permutations + text + ids once per process.
    """
    global _G_PERMS, _G_WINDOW, _G_TEXTS, _G_IDS
    _G_PERMS  = perms
    _G_WINDOW = int(window_size)
    _G_TEXTS  = texts
    _G_IDS    = ids


def _minhash_worker_range(start: int, end: int) -> List[Tuple[int, np.ndarray]]:
    """
    Worker: compute MinHash for rows [start:end) from global arrays.
    Returns [(doc_id, mh_uint32[K]), ...]
    """
    assert _G_PERMS is not None
    assert _G_WINDOW is not None
    assert _G_TEXTS is not None
    assert _G_IDS is not None

    perms = _G_PERMS
    win   = _G_WINDOW
    texts = _G_TEXTS
    ids   = _G_IDS

    out: List[Tuple[int, np.ndarray]] = []
    for idx in range(start, end):
        mh = minhash_signature(texts[idx], perms, win=win)
        out.append((int(ids[idx]), mh))
    return out


def parallel_minhash_fast_milvus(
    ids: np.ndarray,
    texts: np.ndarray,
    perms: np.ndarray,
    batch_size: int,
    *,
    window_size: int,
    n_proc: int,
    verbose: bool = True,
) -> List[Tuple[int, np.ndarray]]:
    """
    Parallel MinHash over a slice of (ids, texts):

      - Uses n_proc worker processes (typically `--threads` from CLI),
        clamped to [1, N].
      - Splits the N rows into ~equal contiguous ranges, one task per worker.
      - Broadcasts perms + texts + ids once per worker via initializer.
      - Returns [(doc_id, mh_uint32[K])].
    """
    N = int(len(ids))
    if N == 0:
        return []

    if len(texts) != N:
        raise ValueError("ids and texts must have the same length")

    # Decide number of workers based on n_proc
    workers = max(1, int(n_proc) if n_proc is not None else 1)
    workers = min(workers, N)



    # Build index ranges for tasks (avoid pickling large arrays)
    bs = max(1, batch_size)
    ranges = [(off, min(off + bs, N)) for off in range(0, N, bs)]
    num_batches = len(ranges)



    if verbose:
        print(
            f"[parallel_minhash_fast_milvus] rows={N} "
            f"workers={workers} batches={num_batches}"
        )

    t0 = time.perf_counter()
    results: List[Tuple[int, np.ndarray]] = []

    if workers == 1 or num_batches == 1:
        # Single-process path, but still use same worker logic for consistency.
        _init_worker_mh(perms, window_size, texts, ids)
        for s, e in ranges:
            results.extend(_minhash_worker_range(s, e))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker_mh,
            initargs=(perms, int(window_size), texts, ids),
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
                        f"[parallel_minhash_fast_milvus] done {done}/{num_batches} "
                        f"range=[{s}:{e}) "
                        f"wait={(time.perf_counter() - t_submit):.3f}s"
                    )

    total_s = time.perf_counter() - t0
    if verbose:
        print(
            f"[parallel_minhash_fast_milvus] total={total_s:.3f}s "
            f"({total_s/60:.2f} min), items={len(results)}"
        )


    # results.sort(key=lambda t: t[0])
    return results

# ---------- CLI / main ----------
def main():
    ap = argparse.ArgumentParser("MINHASH_LSH ingest (batched) using manifests from --outdir")
    ap.add_argument("--outdir", required=True, type=Path,
                    help="Directory that contains manifests/sketch_spec.json and permutations.npy")
    ap.add_argument("--uri", default=os.environ.get("MILVUS_URI", "http://127.0.0.1:19530"))
    ap.add_argument("--collection", default="corpus_minhash_lsh")
    ap.add_argument("--recreate", action="store_true",
                    help="Drop and recreate collection/index before ingest")
    ap.add_argument("--main_parquet", required=True)
    ap.add_argument("--id_col", default="int_id_column")
    ap.add_argument("--text_col", default="contents")
    ap.add_argument("--insert_limit", type=int, default=0,
                    help="Global cap on rows to ingest (0 = no cap)")
    ap.add_argument("--threads", type=int, default=32)

    # batching mh_batch
    ap.add_argument("--batch_rows", type=int, default=10_000,
                    help="Parquet slice size per collect()")
    ap.add_argument("--insert_batch", type=int, default=10_000,
                    help="Rows per Milvus insert() call")

    ap.add_argument("--mh_batch", type=int, default=10_000)


    # Index params (must match how you want to search)
    ap.add_argument("--mh_bit_width", type=int, default=32, choices=[8,16,32,64],
                    help="Per-lane width; vector dim_bits = K * mh_bit_width")
    ap.add_argument("--bands", type=int, default=14, help="mh_lsh_band")
    ap.add_argument("--window", type=int, default=5, help="word-shingle window size")
    ap.add_argument("--partition", type=str, default=None)

    args = ap.parse_args()

    outdir = args.outdir.resolve()
    spec, perms = load_spec_and_perms(outdir)
    K = int(perms.shape[0])
    dim_bits = 112 * int(args.mh_bit_width)

    print(f"Loaded manifests from {outdir}/manifests")
    print(f"  K={K}, mh_bit_width={args.mh_bit_width}  → dim_bits={dim_bits}")
    if "mmh3_seed" in spec:
        print(f"  mmh3_seed (info): {spec['mmh3_seed']}")

    client = MilvusClient(uri=args.uri)

    print(f"Connected → {args.uri}")

    # (Re)create collection/index or append to existing
    if client.has_collection(args.collection) and args.recreate:
        print(f"• Dropping existing collection {args.collection} …")
        client.drop_collection(args.collection)

    if not client.has_collection(args.collection):
        print(f"• Creating collection {args.collection}")
        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="MinHash LSH over word-shingles; token_set keeps lane values (decimals)"
        )
        schema.add_field("doc_id", DataType.INT64, is_primary=True)
        schema.add_field("minhash_signature", DataType.BINARY_VECTOR, dim=dim_bits)
        schema.add_field("token_set", DataType.VARCHAR, max_length=65535)  # for refined Jaccard

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="minhash_signature",
            index_type="MINHASH_LSH",
            metric_type="MHJACCARD",
            params={
                "mh_element_bit_width": int(args.mh_bit_width),
                "mh_lsh_band": int(args.bands),
                "with_raw_data": True,
                "mh_lsh_code_in_mem": True
            },
        )

        client.create_collection(args.collection, schema=schema, index_params=index_params)
        print("• Created collection + index")
    else:
        print(f"• Appending to existing collection {args.collection}")

    client.load_collection(args.collection)

    # Build lazy scan
    ldf = (
        pl.scan_parquet(args.main_parquet)
          .select(pl.col(args.id_col).alias("doc_id"),
                  pl.col(args.text_col).alias("contents"))
          .with_columns(pl.col("doc_id").cast(pl.Int64),
                        pl.col("contents").cast(pl.Utf8))
    )
    if args.insert_limit and args.insert_limit > 0:
        ldf = ldf.limit(args.insert_limit)
    


    total_rows = ldf.select(pl.len()).collect(engine="streaming").item()


    if total_rows == 0:
        print("No rows to ingest. Exiting.")
        return

    expected_bytes = (K * int(args.mh_bit_width)) // 8
    print(f"Planned ingest: {total_rows:,} rows  (batch_rows={args.batch_rows:,}, insert_batch={args.insert_batch:,})")

    # Slice & ingest + timing
    start = 0
    inserted_total = 0
    batch_idx = 0

    per_slice_stats: List[Dict[str, Any]] = []
    total_insert_ms = 0.0

    while start < total_rows:
        stop = min(start + args.batch_rows, total_rows)
        df = ldf.slice(start, stop - start).collect(engine="streaming")
        batch_idx += 1

        # Get ids + texts as numpy arrays for worker broadcast
        ids = np.asarray(df["doc_id"].to_list(), dtype=np.int64)
        texts = np.asarray(df["contents"].to_list(), dtype=object)

        # --- NEW: parallel MinHash over this slice ---
        t_mh0 = time.time()
        tuples = parallel_minhash_fast_milvus(
            ids=ids,
            texts=texts,
            perms=perms,
            batch_size=args.mh_batch,
            window_size=args.window,
            n_proc=args.threads,   # lock to user-specified thread count
            verbose=False,
        )


        mh_ms = (time.time() - t_mh0) * 1000.0
        print(f"[slice {batch_idx}] MinHash computed for {len(tuples):,} rows in {mh_ms:.1f} ms")

        # Build rows_slice exactly like before, but from (doc_id, mh) tuples
        rows_slice: List[Dict[str, Any]] = []
        for doc_id, mh in tuples:
            sig = mh_words_to_bytes(mh, args.mh_bit_width)
            if len(sig) != expected_bytes:
                raise RuntimeError(f"encoded bytes {len(sig)} != expected {expected_bytes}")
            tok = mh_words_to_token_str(mh)
            rows_slice.append({
                "doc_id": int(doc_id),
                "minhash_signature": sig,
                "token_set": tok,
            })

        insert_ms_slice = 0.0
        n_sub_batches = 0

        if rows_slice:
            for i in range(0, len(rows_slice), args.insert_batch):
                sub = rows_slice[i:i + args.insert_batch]
                t0 = time.time()
                client.insert(args.collection, sub, partition_name=args.partition)
                dt_ms = (time.time() - t0) * 1000.0
                insert_ms_slice += dt_ms
                n_sub_batches += 1
                inserted_total += len(sub)

            total_insert_ms += insert_ms_slice
            print(
                f"[slice {batch_idx}] rows {start:,}-{stop-1:,} → inserted {len(rows_slice):,}  "
                f"(sub-batches={n_sub_batches}, insert_ms={insert_ms_slice:.1f})  "
                f"(total={inserted_total:,})"
            )

        # record per-slice timing
        per_slice_stats.append({
            "slice": batch_idx,
            "start_row": int(start),
            "end_row": int(stop - 1),
            "n_rows": int(len(rows_slice)),
            "n_sub_batches": int(n_sub_batches),
            "insert_ms": float(insert_ms_slice),
            "avg_ms_per_1k_rows": (insert_ms_slice / max(1, len(rows_slice)) * 1000.0) if rows_slice else None
        })

        start = stop

    # client.flush(args.collection)

    # Final load to make sure new segments are searchable
    client.load_collection(args.collection)
    print(f"✓ Ingest complete. Total inserted this run: {inserted_total:,}")
    print(f"   Total insert time: {total_insert_ms:.1f} ms")

    # ---------- Save metrics ----------
    metrics_dir = outdir / "metrics"
    ensure_dir(metrics_dir)
    out_path = metrics_dir / f"ingest_insert_times_{args.collection}.json"
    summary = {
        "collection": args.collection,
        "main_parquet": str(Path(args.main_parquet).resolve()),
        "total_rows_planned": int(total_rows),
        "inserted_total": int(inserted_total),
        "batch_rows": int(args.batch_rows),
        "insert_batch": int(args.insert_batch),
        "mh_bit_width": int(args.mh_bit_width),
        "bands": int(args.bands),
        "window": int(args.window),
        "partition": args.partition,
        "K": int(K),
        "dim_bits": int(dim_bits),
        "total_insert_ms": float(total_insert_ms),
        "avg_insert_ms_per_row": (total_insert_ms / max(1, inserted_total)),
        "avg_insert_ms_per_1k_rows": (total_insert_ms / max(1, inserted_total) * 1000.0),
        "slices": per_slice_stats,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Wrote insert timing metrics → {out_path}")
    print(f"   Manifests used: {outdir/'manifests'}")

if __name__ == "__main__":
    main()
