#!/usr/bin/env python3
from __future__ import annotations

import argparse, os, time, json, threading
from pathlib import Path
from typing import List, Dict, Any, Sequence, Tuple, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
import pandas as pd

import faiss
import mmh3
import psutil
import subprocess
import sys
import tempfile

# ============================================================
# Filesystem helpers
# ============================================================

def ensure_dir(p: Path) -> None:
    """Create directory (and parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> Any:
    """Read JSON file from path."""
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(p: Path, obj: Any) -> None:
    """Write JSON file with indentation."""
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def file_size(p: Path) -> int:
    """Return size of file in bytes, or 0 on failure."""
    try:
        return p.stat().st_size
    except Exception:
        return 0

def dir_size_bytes(root: Path) -> int:
    """Total size (in bytes) of all files under a directory tree."""
    total = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            try:
                total += (Path(dp) / fn).stat().st_size
            except Exception:
                pass
    return total

# ============================================================
# Memory snapshots (system & process)
# ============================================================

def system_mem_snapshot() -> dict:
    """
    System-wide memory snapshot (roughly matches htop 'Mem' line).
    """
    vm = psutil.virtual_memory()
    total = int(vm.total)
    used  = int(vm.used)          # includes cache+buffers
    avail = int(vm.available)
    free  = int(getattr(vm, "free", 0) or 0)
    buffers = int(getattr(vm, "buffers", 0) or 0)
    cached  = int(getattr(vm, "cached", 0) or 0)
    used_no_cache = total - (free + buffers + cached)
    return {
        "sys_total_bytes": total,
        "sys_used_bytes": used,
        "sys_available_bytes": avail,
        "sys_used_no_cache_bytes": int(max(0, used_no_cache)),
        "sys_used_percent": float(vm.percent),
        "sys_buffers_bytes": buffers,
        "sys_cached_bytes": cached,
        "sys_free_bytes": free,
        "ts": time.time(),
        "ts_iso": datetime.utcnow().isoformat() + "Z",
    }

def process_tree_rss_bytes() -> int:
    """
    RSS of current process + all children (best-effort).
    Similar to htop when expanding a process tree.
    """
    try:
        root = psutil.Process(os.getpid())
        procs = [root] + root.children(recursive=True)
        rss = 0
        for p in procs:
            try:
                rss += p.memory_info().rss
            except Exception:
                pass
        return rss
    except Exception:
        return 0

# ============================================================
# System sampler (periodic CPU% & memory)
# ============================================================

class SystemSampler:
    """
    Background thread that samples CPU% and system memory every N seconds.
    Used to record coarse-grained resource usage during index build.
    """
    def __init__(self, interval_sec: float = 60.0):
        self.interval = float(interval_sec)
        self._stop = threading.Event()
        self.samples: List[Dict[str, Any]] = []
        self._thr: Optional[threading.Thread] = None
        # warm-up so first cpu_percent call is meaningful
        psutil.cpu_percent(interval=None)

    def start(self) -> None:
        """Start sampler thread if not already running."""
        if self._thr and self._thr.is_alive():
            return
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self) -> None:
        """Stop sampler thread and join."""
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=self.interval + 2.0)

    def _run(self) -> None:
        """Internal loop: sleep for interval, record CPU and mem."""
        while not self._stop.is_set():
            cpu = float(psutil.cpu_percent(interval=self.interval))
            mem = system_mem_snapshot()
            mem["cpu_percent_avg"] = cpu
            self.samples.append(mem)

    def summary(self) -> Dict[str, Any]:
        """Aggregate statistics from collected samples."""
        if not self.samples:
            mem = system_mem_snapshot()
            return {
                "n_samples": 0,
                "cpu_avg_percent": None,
                "cpu_peak_percent": None,
                "sys_peak_used_bytes": mem["sys_used_bytes"],
                "sys_peak_used_no_cache_bytes": mem["sys_used_no_cache_bytes"],
                "sys_total_bytes": mem["sys_total_bytes"],
            }
        cpu_vals = [s["cpu_percent_avg"] for s in self.samples]
        used_vals = [s["sys_used_bytes"] for s in self.samples]
        used_nc_vals = [s["sys_used_no_cache_bytes"] for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "cpu_avg_percent": float(np.mean(cpu_vals)),
            "cpu_peak_percent": float(np.max(cpu_vals)),
            "sys_peak_used_bytes": int(np.max(used_vals)),
            "sys_peak_used_no_cache_bytes": int(np.max(used_nc_vals)),
            "sys_total_bytes": int(self.samples[-1]["sys_total_bytes"]),
        }

# ============================================================
# Manifests (sketch spec + permutations)
# ============================================================

def load_spec_and_perms(outdir: Path):
    """Load sketch_spec.json and permutations.npy from manifests/."""
    spec  = read_json(outdir / "manifests" / "sketch_spec.json")
    perms = np.load(outdir / "manifests" / "permutations.npy")
    return spec, perms

# ============================================================
# Bitmaps & MinHash helpers
# ============================================================

def minhash_to_bitmap_mmh3(
    minhashes: np.ndarray,
    M: int,
    seed: int,
    endian: str = "little",
) -> np.ndarray:
    """
    Convert a MinHash vector into an M-bit bitmap via mmh3.

    minhashes: np.uint32[K]
    return:    np.uint64[(M+63)//64] bitmap (packed 64-bit words)
    """
    words = np.zeros((M + 63) // 64, dtype=np.uint64)
    pow2 = (M & (M - 1)) == 0
    mask = (M - 1) if pow2 else None
    for v in np.asarray(minhashes, dtype=np.uint32):
        x = int(v)
        b = x.to_bytes(4, endian)
        h = mmh3.hash64(b, seed=seed, signed=False)[0]
        idx = (h & mask) if pow2 else (h % M)
        words[idx >> 6] |= (np.uint64(1) << np.uint64(idx & 63))
    return words

def _generate_minhash_blocked(
    permutations: np.ndarray,
    text: str,
    *,
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
    K = int(permutations.shape[0])
    if n == 0:
        return np.zeros((K,), dtype=np.uint32)

    # Build k-shingles with sliding window (ensure ≥1 shingle)
    shingles = [
        delimiter.join(words[i:i + window_size])
        for i in range(max(1, n - window_size + 1))
    ]
    hv = np.array([mmh3.hash(s, signed=False) for s in shingles], dtype=np.uint64)

    a = permutations.astype(np.uint64, copy=False)
    best = np.full((K,), np.uint64((1 << 64) - 1), dtype=np.uint64)

    for start in range(0, hv.shape[0], CH):
        # hv block → multiply-shift → take minimum per permutation
        sl = hv[start:start + CH][:, None]          # (CH, 1)
        vals = (sl * a[None, :]) >> np.uint64(32)   # (CH, K)
        local = vals.min(axis=0)
        np.minimum(best, local, out=best)

    return best.astype(np.uint32, copy=False)




# =========================
# Range-based MinHash (same pattern as query script)
# =========================

# Worker globals for build-side MinHash
_G_PERM:   np.ndarray | None = None
_G_WINDOW: int | None = None
_G_CHUNK:  int | None = None
_G_CONTENTS: np.ndarray | None = None  # np.ndarray of str
_G_IDS:      np.ndarray | None = None  # np.ndarray of int


def _init_worker_build(
    permutations: np.ndarray,
    window_size: int,
    block_CH: int,
    contents: np.ndarray,
    ids: np.ndarray,
):
    """
    Initializer for build-side workers: broadcast permutations + text arrays once.
    """
    global _G_PERM, _G_WINDOW, _G_CHUNK, _G_CONTENTS, _G_IDS
    _G_PERM     = permutations
    _G_WINDOW   = int(window_size)
    _G_CHUNK    = int(block_CH)
    _G_CONTENTS = contents
    _G_IDS      = ids


def _minhash_worker_fast_range_build(start: int, end: int) -> List[Tuple[int, np.ndarray]]:
    """
    Worker: compute MinHash for rows [start:end) using globals.
    Returns: List[(doc_id:int, mh:np.ndarray[uint32, K])]
    """
    assert _G_PERM is not None
    assert _G_WINDOW is not None
    assert _G_CONTENTS is not None
    assert _G_IDS is not None

    perms = _G_PERM
    wsize = _G_WINDOW
    CH    = _G_CHUNK or 8192
    C     = _G_CONTENTS
    I     = _G_IDS

    out: List[Tuple[int, np.ndarray]] = []
    for idx in range(start, end):
        # blocked MinHash kernel (same as query script)
        mh = _generate_minhash_blocked(perms, C[idx], window_size=wsize, CH=CH)
        out.append((int(I[idx]), mh))
    return out

def parallel_minhash_fast_build(
    df_pd: pd.DataFrame,
    permutations: np.ndarray,
    *,
    window_size: int,
    n_proc: int,
    block_CH: int = 8192,
    verbose: bool = True,
    contents_col: str = "contents",
    id_col: str = "doc_id",
) -> List[Tuple[int, np.ndarray]]:
    """
    Build-side parallel MinHash using all requested threads:

      - Use n_proc (typically `threads` from CLI) as the number of worker
        processes (clamped to [1, N]).
      - Split the N rows into ~equal contiguous ranges, one task per worker.
      - Returns [(doc_id:int, mh:np.ndarray[uint32,K])] sorted by doc_id.
    """
    N = len(df_pd)
    if N == 0:
        return []

    # Single extraction of columns; on Linux these are COW after fork.
    contents = df_pd[contents_col].values
    ids      = df_pd[id_col].values

    # Decide #workers from n_proc (=threads), but never more than N rows.
    # workers = max(1, int(n_proc) if n_proc is not None else 1)
    # workers = min(workers, N)
    workers = n_proc

    # Build contiguous ranges that cover all rows
    ranges: List[Tuple[int, int]] = []
    base = N // workers
    rem  = N % workers
    start = 0
    for w in range(workers):
        end = start + base + (1 if w < rem else 0)
        if start >= end:
            break
        ranges.append((start, end))
        start = end
    num_batches = len(ranges)

    if verbose:
        print(
            f"[parallel_minhash_fast_build] rows={N} workers={workers} "
            f"batches={num_batches}"
        )

    t0 = time.perf_counter()
    results: List[Tuple[int, np.ndarray]] = []

    if workers == 1 or num_batches == 1:
        # Single-process path: still use the same worker logic
        _init_worker_build(permutations, window_size, block_CH, contents, ids)
        for s, e in ranges:
            results.extend(_minhash_worker_fast_range_build(s, e))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker_build,
            initargs=(permutations, int(window_size), int(block_CH), contents, ids),
        ) as pool:
            fut_meta: Dict[Any, Tuple[int, int, int, float]] = {}
            for i, (s, e) in enumerate(ranges, 1):
                t_submit = time.perf_counter()
                fut = pool.submit(_minhash_worker_fast_range_build, s, e)
                fut_meta[fut] = (i, s, e, t_submit)

            done = 0
            for fut in as_completed(fut_meta):
                i, s, e, t_submit = fut_meta[fut]
                try:
                    chunk = fut.result()
                except Exception as ex:
                    raise RuntimeError(
                        f"_minhash_worker_fast_range_build failed for [{s}:{e})"
                    ) from ex
                results.extend(chunk)
                done += 1
                if verbose:
                    print(
                        f"[parallel_minhash_fast_build] done {done}/{num_batches} "
                        f"range=[{s}:{e}) wait={(time.perf_counter() - t_submit):.3f}s"
                    )

    total_s = time.perf_counter() - t0
    if verbose:
        print(
            f"[parallel_minhash_fast_build] total={total_s:.3f}s "
            f"({total_s/60:.2f} min), items={len(results)}"
        )

    # Keep stable order by doc_id (same as query side)
    results.sort(key=lambda t: t[0])
    return results



def _generate_minhash_join(
    permutations: np.ndarray,
    text: str,
    *,
    window_size: int = 5,
    delimiter: str = " ",
    CH: int = 8192,
) -> np.ndarray:
    """
    Original MinHash path (join-based shingles), kept for parity/testing.
    """
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
    a  = permutations.astype(np.uint64, copy=False)
    best = np.full((K,), np.uint64((1 << 64) - 1), dtype=np.uint64)
    for start in range(0, hv.shape[0], CH):
        sl = hv[start:start + CH][:, None]
        vals = (sl * a[None, :]) >> np.uint64(32)
        local = vals.min(axis=0)
        np.minimum(best, local, out=best)
    return best.astype(np.uint32, copy=False)

# ============================================================
# Worker globals & routines (used by ProcessPoolExecutor)
# ============================================================

# Shared config for workers (broadcast once per process)
_G: Dict[str, Any] = {
    "PERM": None,
    "WIN": None,
    "MODE": None,
    "M": None,
    "SEED": None,
    "BLOCK_CH": 8192,
    "USE_BLOCKED": True,
}





def tuples_to_items_vecs(
    tuples_list: List[Tuple[int, np.ndarray]],
    M_bits: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Convert [(doc_id, mh_uint32[K])] -> [{"doc_id", "vectors"}] with 4096-bit bitmaps.
    """
    from numpy import uint8
    out: List[Dict[str, Any]] = []
    for doc_id, mh in tuples_list:
        vflat = np.array(mh, dtype=np.uint32)
        vec   = vflat.astype(">u4").view(np.uint8).reshape(1, -1)
        out.append({"doc_id": int(doc_id), "vectors": vec})
    return out

# ============================================================
# Polars streaming over Parquet
# ============================================================

def _rows_iter_pl(ldf: pl.LazyFrame, total: int, batch_rows: int):
    """
    Yield small Polars DataFrames by slicing a LazyFrame in row chunks.
    """
    start = 0
    while start < total:
        stop = min(start + batch_rows, total)
        yield ldf.slice(start, stop - start).collect(engine="streaming")
        start = stop


def run_minhash_helper_batch_df(
    *,
    helper_script: Path,
    outdir: Path,
    df_pl: pl.DataFrame,
    id_col: str,
    text_col: str,
    window: int,
    threads: int,
    block_CH: int = 8192,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Write df_pl to a temporary parquet, call the helper script,
    and load (XB, ids, mh_ms) from its .npz output.

    Returns:
      XB    : np.uint8 [N, M_bits/8]
      ids   : np.int64 [N]
      mh_ms : float (MinHash-only time in ms, measured inside helper)
    """
    import numpy as np

    helper_script = Path(helper_script)
    if not helper_script.exists():
        raise FileNotFoundError(f"MinHash helper script not found: {helper_script}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        batch_parquet = tmpdir / "batch.parquet"
        out_npz = tmpdir / "batch_mh.npz"

      
        df_pl.write_parquet(batch_parquet)

        cmd = [
            sys.executable,
            str(helper_script),
            "--outdir", str(outdir),
            "--batch_parquet", str(batch_parquet),
            "--id_col", "doc_id",
            "--text_col", "contents",
            "--window", str(window),
            "--threads", str(threads),
            "--block_CH", str(block_CH),
            "--out_npz", str(out_npz),
        ]
        subprocess.run(cmd, check=True)

        data = np.load(out_npz)
        XB    = data["xb"]
        ids   = data["ids"]
        mh_ms = float(data["mh_ms"])
        return XB, ids, mh_ms

# ============================================================
# Build / Append HNSW index
# ============================================================

def build_or_append_index(
    *,
    outdir: Path,
    series: str,
    metric: str,        # "hamming" | "jaccard" | "minhash_hamming"
    M_hnsw: int,
    corpus_parquet: Path,
    id_col: str,
    text_col: str,
    permutations: np.ndarray,
    M_bits: int,
    K: int,
    mmh3_seed: int,
    window: int,
    efC: int,
    threads: int,
    mh_batch: int,
    add_batch: int,
    max_rows: Optional[int],
    prev_index_dir: Optional[Path],
    sys_sample_sec: float,
) -> None:
    """
    Main build/append pipeline:
      - Stream corpus from Parquet via Polars.
      - MinHash + bitmap for each batch (parallel).
      - Insert into IndexBinaryHNSW.
      - Track resource usage and write metadata.
    """

    faiss.omp_set_num_threads(threads)

    # We always use bitmap vectors for HNSW here.
    vec_mode = "bitmap"
    d_bits = int(M_bits)
    d_bits = 112*32
    metric_tag = metric  # label used in directory layout

    # Output directory layout
    idx_dir = outdir / "indices" / series / metric_tag / f"M{M_hnsw}"
    ensure_dir(idx_dir)
    idx_path = idx_dir / "index.faiss"
    labels_path = idx_dir / "labels.npy"
    meta_path = idx_dir / "index.json"

    # --------------------------------------------------------
    # Load existing index (append) or create a new one
    # --------------------------------------------------------
    appending = False
    if prev_index_dir is not None:
        # Append to explicit previous index directory
        p_idx = prev_index_dir / "index.faiss"
        p_lab = prev_index_dir / "labels.npy"
        if not p_idx.exists() or not p_lab.exists():
            raise FileNotFoundError(
                f"--prev_index_dir missing index or labels: {prev_index_dir}"
            )
        index = faiss.read_index_binary(str(p_idx))
        if not hasattr(index, "hnsw"):
            raise RuntimeError("prev index not IndexBinaryHNSW")
        labels = np.load(p_lab).astype(np.int64, copy=False)
        if index.ntotal != labels.shape[0]:
            raise RuntimeError("prev labels.npy length mismatch")
        if getattr(index, "d", None) and int(index.d) != int(d_bits):
            raise RuntimeError(f"Binary dim mismatch: loaded {index.d} vs expected {d_bits}")
        appending = True
        print(f"• Appending to existing index ({prev_index_dir}), ntotal={index.ntotal:,}")
    elif idx_path.exists() and labels_path.exists():
        # Append if default index already exists
        index = faiss.read_index_binary(str(idx_path))
        if not hasattr(index, "hnsw"):
            raise RuntimeError("existing index not IndexBinaryHNSW")
        labels = np.load(labels_path).astype(np.int64, copy=False)
        if index.ntotal != labels.shape[0]:
            raise RuntimeError("labels.npy length mismatch")
        if getattr(index, "d", None) and int(index.d) != int(d_bits):
            raise RuntimeError(f"Binary dim mismatch: loaded {index.d} vs expected {d_bits}")
        appending = True
        print(f"• Appending to existing index ({idx_dir}), ntotal={index.ntotal:,}")
    else:
        # Build brand new index
        index = faiss.IndexBinaryHNSW(int(d_bits), int(M_hnsw))
        index.hnsw.efConstruction = int(efC)
        index.hnsw.efSearch = 32
        labels = np.empty((0,), dtype=np.int64)
        print(f"• Building NEW index (M={M_hnsw}, d_bits={d_bits}, mode={vec_mode})")

    # Set HNSW parameters for this run
    index.hnsw.efConstruction = int(efC)
    index.hnsw.efSearch = 16

    if metric == "jaccard":
        print("• Metric tag: jaccard (directory label). Using binary HNSW core.")

    # --------------------------------------------------------
    # Stream corpus via Polars
    # --------------------------------------------------------
    ldf = (
        pl.scan_parquet(corpus_parquet)
          .select(
              pl.col(id_col).alias("doc_id"),
              pl.col(text_col).alias("contents"),
          )
          .with_columns(
              pl.col("doc_id").cast(pl.Int64),
              pl.col("contents").cast(pl.Utf8),
          )
    )
    total_rows = ldf.select(pl.len()).collect(engine="streaming").item()
    if max_rows is not None:
        total_rows = min(total_rows, int(max_rows))
    print(
        f"• Ingesting {total_rows:,} rows from {corpus_parquet}  "
        f"({'append' if appending else 'new build'})"
    )

    # Start system sampler (CPU% & memory every sys_sample_sec seconds)
    sampler = SystemSampler(interval_sec=sys_sample_sec)
    sampler.start()

    added = 0
    ids_acc: List[np.ndarray] = []
    t0 = time.time()

    peak_rss = 0
    batch_stats: List[Dict[str, Any]] = []
    batch_idx = 0
    mh_ms_total = 0.0
    add_ms_total = 0.0


    helper_script = Path(__file__).with_name("rad_minhash_helper_batch.py")

    for df_pl in _rows_iter_pl(ldf, total_rows, add_batch):
        if max_rows is not None and added >= max_rows:
            break

        df_pd = df_pl.to_pandas()





        # ---------- MinHash + vectorization (legacy path for comparability) ----------
        t_mh0 = time.time()
        t_mh1 = time.perf_counter()



        tuples = parallel_minhash_fast(
            df_pd,
            permutations,
            batch_size=mh_batch,          # same mh_batch as CLI
            window_size=window,
            n_proc=threads,               # still passed, but legacy code caps at 25
            block_CH=8192,
            verbose=False,
            contents_col="contents",
            id_col="doc_id",              # build-side ID column
        )

        mh_ms = (time.time() - t_mh0) * 1000.0
        mh_s = time.perf_counter() - t_mh1
        print(f"[MinHash helper] rows={len(df_pd)} mh={mh_ms:.1f}ms (wall {mh_s:.3f} s)")


        items = tuples_to_items_vecs(tuples, M_bits=M_bits, seed=mmh3_seed)
        vecs = [it["vectors"] for it in items]
        ids  = [it["doc_id"]  for it in items]
        

        # Build XB: stack bitmaps into a single uint8 array
        XB = np.vstack(vecs).astype(np.uint8, copy=False)
        print(XB.shape[1])
        assert XB.shape[1] * 8 == M_bits      # e.g. 512 * 8 = 4096

    
        # Rebuild XB explicitly (kept for parity with previous tests)
        XB = np.array(vecs, dtype=np.uint8)
        XB = np.vstack(XB)

        dim_bytes = XB.shape[1]
        bytes_per_hash = dim_bytes // 112
        print("bytes_per_hash:", bytes_per_hash)
        if bytes_per_hash == 4:
            print("Looks like 32-bit MinHash packing.")
        elif bytes_per_hash == 8:
            print("Looks like 64-bit MinHash packing.")
        else:
            print("Weird packing:", bytes_per_hash, "bytes per hash")



        I64 = np.asarray(ids, dtype=np.int64)

        index.hnsw.efConstruction = int(efC)
        index.hnsw.efSearch = 32

        # print(index.hnsw.code_size)
        print(index.code_size)

    
        # ---------- Add to FAISS index ----------
        t_add0 = time.time()
        index.add(XB)
        add_ms = (time.time() - t_add0) * 1000.0

        ids_acc.append(I64)
        added += XB.shape[0]

        rss_now = process_tree_rss_bytes()
        peak_rss = max(peak_rss, rss_now)

        sys_now = system_mem_snapshot()

        batch_stats.append({
            "batch_idx": batch_idx,
            "n": int(XB.shape[0]),
            "mh_ms": float(mh_ms),           # from helper
            "add_ms": float(add_ms),
            "rss_bytes": int(rss_now),
            "sys_used_bytes": int(sys_now["sys_used_bytes"]),
            "sys_used_no_cache_bytes": int(sys_now["sys_used_no_cache_bytes"]),
            "cpu_percent_minute_avg": None,
            "ts": sys_now["ts"],
            "ts_iso": sys_now["ts_iso"],
        })
        batch_idx += 1
        mh_ms_total += mh_ms
        add_ms_total += add_ms

        print(
            f"  + {XB.shape[0]:,} (total added {added:,})  "
            f"mh={mh_ms:.1f}ms add={add_ms:.1f}ms "
            f"proc_rss~{rss_now/1e9:.2f}GB "
            f"sys_used~{sys_now['sys_used_bytes']/1e9:.2f}GB "
            f"(apps≈{sys_now['sys_used_no_cache_bytes']/1e9:.2f}GB)"
        )

        if max_rows is not None and added >= max_rows:
            break


    # --------------------------------------------------------
    # Finalize: stop sampler, save index, labels, and metadata
    # --------------------------------------------------------
    sampler.stop()
    sys_summary = sampler.summary()
    sys_timeseries = sampler.samples  # list of per-sample dicts

    # Save index & labels atomically
    new_labels = np.concatenate([labels] + ids_acc) if ids_acc else labels
    tmp_idx = idx_path.with_suffix(".faiss.tmp")
    tmp_lab = labels_path.with_suffix(".tmp.npy")
    faiss.write_index_binary(index, str(tmp_idx))
    np.save(tmp_lab, new_labels.astype(np.int64, copy=False))
    os.replace(tmp_idx, idx_path)
    os.replace(tmp_lab, labels_path)

    t1 = time.time()
    wall_s = (t1 - t0)
    overall_throughput = (added / wall_s) if wall_s > 0 else None

    idx_bytes = file_size(idx_path)
    lab_bytes = file_size(labels_path)
    dir_bytes = dir_size_bytes(idx_dir)

    # Write system timeseries CSV for quick plotting
    ts_csv = idx_dir / "build_resources_timeseries.csv"
    try:
        import csv
        ensure_dir(ts_csv.parent)
        with open(ts_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts_iso",
                "ts",
                "cpu_percent_avg",
                "sys_used_bytes",
                "sys_used_no_cache_bytes",
                "sys_total_bytes",
            ])
            for s in sys_timeseries:
                w.writerow([
                    s["ts_iso"],
                    s["ts"],
                    s.get("cpu_percent_avg", None),
                    s["sys_used_bytes"],
                    s["sys_used_no_cache_bytes"],
                    s["sys_total_bytes"],
                ])
        print(f"✓ Wrote system timeseries CSV → {ts_csv}")
    except Exception as e:
        print(f"⚠️ Could not write timeseries CSV: {e}")

    build_stats = {
        "total_rows": int(total_rows),
        "added_total": int(added),
        "wall_s": float(wall_s),
        "overall_docs_per_s": float(overall_throughput)
        if overall_throughput is not None
        else None,
        "mh_ms_total": float(mh_ms_total),
        "add_ms_total": float(add_ms_total),
        "peak_rss_bytes": int(peak_rss),
        "index_bytes": int(idx_bytes),
        "labels_bytes": int(lab_bytes),
        "dir_bytes": int(dir_bytes),
        "batches": batch_stats,
    }

    meta = {
        "series": series,
        "metric": metric_tag,
        "vec_mode": vec_mode,
        "d_bits": int(d_bits),
        "M_bits": int(M_bits),
        "K": int(K),
        "M_hnsw": int(M_hnsw),
        "efConstruction": int(efC),
        "built_at": datetime.utcnow().isoformat() + "Z",
        "added_this_run": int(added),
        "ntotal_after": int(index.ntotal),
        "labels_count": int(new_labels.shape[0]),
        "mode": "append" if appending else "new",
        "build_stats": build_stats,
        "system_summary": sys_summary,
        "system_timeseries": sys_timeseries,
    }
    write_json(meta_path, meta)
    print(f"✓ Saved index  → {idx_path}")
    print(f"✓ Saved labels → {labels_path}  (ntotal={index.ntotal:,})")
    print(f"✓ Metadata     → {meta_path}")
    print(f"⏱️  Time: {wall_s:.1f}s")

# ============================================================
# CLI entrypoint
# ============================================================


# ============================================================
# Legacy parallel MinHash (same as old query script)
# ============================================================

_G_PERMS:   np.ndarray | None = None
_G_WINDOW:  int | None = None
_G_CHUNK:   int | None = None
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


def parallel_minhash_fast(
    df_pd: pd.DataFrame,
    permutations: np.ndarray,
    *,
    batch_size: int,
    window_size: int,
    n_proc: int,
    block_CH: int = 8192,
    verbose: bool = True,
    return_items: bool = False,  # kept for parity with old version
    contents_col: str = "contents",
    id_col: str = "int_id_column",
) -> List[Tuple[int, np.ndarray]] | List[Dict[str, Any]]:
    """
    Legacy MinHash parallel driver (same as old experiment code).

    Returns by default:
        [(doc_id: int, mh: np.ndarray[uint32, K]), ...] sorted by doc_id.

    If return_items=True, returns:
        [{"minhashes": mh, id_col: doc_id}, ...]
    """
    N = len(df_pd)
    if N == 0:
        return []

    # Extract columns once; on Linux these are COW under fork.
    contents = df_pd[contents_col].values
    ids      = df_pd[id_col].values

    # Build index ranges for tasks (avoid pickling large arrays)
    bs = max(1, batch_size)
    ranges = [(off, min(off + bs, N)) for off in range(0, N, bs)]
    num_batches = len(ranges)



    # Workers chosen as in the old script
    # workers = max(1, min((n_proc or (os.cpu_count() or 8)), num_batches))
    workers = n_proc  # fixed worker cap, matches previous experiments



    if verbose:
        print(
            f"[parallel_minhash_fast] rows={N} batch_size={bs} "
            f"batches={num_batches} workers={workers}"
        )

    t0 = time.perf_counter()
    results: List[Tuple[int, np.ndarray]] = []

    # Single-process path (keeps behavior identical to worker function)
    if workers == 1 or num_batches == 1:
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
            {"minhashes": mh, id_col: int(doc_id)}
            for (doc_id, mh) in results
        ]
    return results


def main() -> None:
    ap = argparse.ArgumentParser("build_index_simple")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--corpus_parquet", required=True, type=Path)
    ap.add_argument("--id_col", default="doc_id")
    ap.add_argument("--text_col", default="contents")
    ap.add_argument("--series", default="1M")
    ap.add_argument(
        "--metric",
        default="hamming",
        choices=["hamming", "jaccard", "minhash_hamming"],
    )
    ap.add_argument("--M", type=int, required=True)

    # Pipeline knobs
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--mh_batch", type=int, default=10_000)
    ap.add_argument("--add_batch", type=int, default=10_000)
    ap.add_argument("--efC", type=int, default=300)

    # Limits
    ap.add_argument("--max_rows", type=int, default=None)

    # Append to previous run
    ap.add_argument("--prev_index_dir", type=Path, default=None)

    # System sampling interval
    ap.add_argument(
        "--sys_sample_sec",
        type=float,
        default=60.0,
        help="Sample CPU% and system memory every N seconds (default: 60)",
    )

    args = ap.parse_args()

    outdir = args.outdir.resolve()
    ensure_dir(outdir / "indices")
    spec, perms = load_spec_and_perms(outdir)
    M_bits  = int(spec["M_bits"])   # bitmap dimension
    K       = int(spec["K"])        # number of MinHash values
    mmh3_sd = int(spec["mmh3_seed"])

    # Scale efConstruction based on M (kept as tested)
    efC = int(args.M) * 4


    local_metric = args.metric

    build_or_append_index(
        outdir=outdir,
        series=args.series,
        metric=local_metric,
        M_hnsw=args.M,
        corpus_parquet=args.corpus_parquet,
        id_col=args.id_col,
        text_col=args.text_col,
        permutations=perms,
        M_bits=M_bits,
        K=K,
        mmh3_seed=mmh3_sd,
        window=args.window,
        efC=efC,
        threads=args.threads,
        mh_batch=args.mh_batch,
        add_batch=args.add_batch,
        max_rows=args.max_rows,
        prev_index_dir=args.prev_index_dir,
        sys_sample_sec=args.sys_sample_sec,
    )

if __name__ == "__main__":
    main()
