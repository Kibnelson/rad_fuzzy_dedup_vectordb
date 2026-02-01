#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json, math, shutil
from pathlib import Path
import polars as pl

# Optional: your existing deps (Ray/Spark dedup impls)
try:
    from data_processing.utils import ParamsUtils  # noqa: F401
    from dpk_fdedup.transform_python import parse_args  # noqa: F401
    import time  # noqa: F401
    from spark_local import SparkServiceOrchestrator  # noqa: F401
    from ray_local import FdedupLocal
    from pathlib import Path as _Path  # noqa: F401
    import pathlib  # noqa: F401
    from typing import Iterable, Dict, Any, Callable  # noqa: F401
    print("ray_local available")
    _HAS_RAY_LOCAL = True
except Exception:
    print("  ray_local NOT available; cannot run FdedupLocal()")
    _HAS_RAY_LOCAL = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def split_parquet(df_path: Path, out_dir: Path, num_workers: int, prefix: str = "part"):
    ensure_dir(out_dir)
    ldf = pl.scan_parquet(df_path)
    n = ldf.select(pl.len()).collect(streaming=True).item()
    if n == 0:
        return []
    sz = math.ceil(n / max(1, num_workers))
    parts = []
    start = 0
    while start < n:
        stop = min(start + sz, n)
        chunk = ldf.slice(start, stop - start).collect(streaming=True)
        p = out_dir / f"{prefix}_{len(parts)}.parquet"
        chunk.write_parquet(p)
        parts.append(p)
        start = stop
    return parts


def merge_cleaned_parquets(folder: Path, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder}")
    lf = pl.concat([pl.scan_parquet(p) for p in files], how="vertical")
    lf.collect(streaming=True).write_parquet(out_path, compression="zstd")
    return out_path


def run_ray_dedup(work_root: Path, input_parquet: Path, id_col: str, text_col: str,
                  num_workers: int, thr: float):
    if not _HAS_RAY_LOCAL:
        raise RuntimeError("ray_local.FdedupLocal not found in env.")
    inp = work_root / "input"
    out = work_root / "output"
    ensure_dir(inp)
    ensure_dir(out)
    shards = split_parquet(input_parquet, inp, num_workers=num_workers)
    if not shards:
        raise RuntimeError("Empty input parquet for dedup.")

    FdedupLocal(
        input_folder=str(inp) + "/",
        output_folder=str(out) + "/",
        contents_column=text_col,
        document_id_column=id_col,
        num_permutations=112,
        num_bands=14,
        num_minhashes_per_band=8,
        jaccard_similarity_threshold=thr,
        shingle_option="word",
        services="SignatureCalculation,ClusterAnalysis,GetDuplicateList,DataCleaning",
        operation_mode="filter_duplicates",
        run_locally=True,
    ).transform()

    cleaned = out / "cleaned"
    return cleaned if cleaned.exists() else out


def derive_output_paths(outdir: Path, input_parquet: Path | None, file_name: str | None):
    """
    Decide output filenames:
      base = --file_name if given, else stem of input_parquet, else 'corpus'
      parquet -> cache/{base}_dedup.parquet
      manifest -> manifests/{base}_dedup.json
    """
    base = file_name or (input_parquet.stem if input_parquet is not None else "corpus")
    dedup_parquet = outdir / "cache" / f"{base}_dedup.parquet"
    manifest_json = outdir / "manifests" / f"{base}_dedup.json"
    return base, dedup_parquet, manifest_json

from typing import Optional, List
from pathlib import Path

def resolve_input_parquet(outdir: Path, corpus_cache_arg: Optional[Path]) -> Path:
    """
    Priority:
      1) --corpus_cache if provided
      2) Most-recent non-dedup file matching cache/corpus*.parquet
      3) Most-recent non-dedup *.parquet in cache/
    """
    if corpus_cache_arg:
        return corpus_cache_arg.resolve()

    cdir = (outdir / "cache").resolve()

    # Prefer files starting with "corpus" and not already deduped
    candidates = [p for p in cdir.glob("corpus*.parquet") if not p.name.endswith("_dedup.parquet")]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # Fallback: any non-dedup parquet in cache/
    candidates = [p for p in cdir.glob("*.parquet") if not p.name.endswith("_dedup.parquet")]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        f"No suitable input parquet found under {cdir}. "
        f"Pass --corpus_cache EXPLICITLY if needed."
    )



def clean_folder_except(
    folder_path: Union[str, Path],
    exclude: Union[str, Iterable[str]] = "docs_to_remove_consolidated",
    *,
    verbose: bool = True,
) -> None:
    """
    Remove everything under `folder_path` EXCEPT the directory/ies whose
    basename matches `exclude` *and their contents*. Ancestor dirs stay,
    but siblings of an excluded dir are still removed.
    """
    root = Path(folder_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    excl_names: List[str] = (
        [exclude]
        if isinstance(exclude, (str, bytes, Path))
        else list(exclude)
    )

    # find every excluded directory
    excluded_dirs = [
        p for p in root.rglob("*") if p.is_dir() and p.name in excl_names
    ]

    if not excluded_dirs and verbose:
        print("  No matching exclude directory found.")

    def is_protected(path: Path) -> bool:
        """Keep path if it is (a) an excluded dir, (b) inside one, or (c) an ancestor."""
        for ex in excluded_dirs:
            if path == ex or ex in path.parents:      # ex or inside ex
                return True
            if path in ex.parents:                    # ancestor of ex
                return True
        return False

    # walk bottom-up so children go first
    removed = kept = 0
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if is_protected(path):
            kept += 1
            if verbose and path in excluded_dirs:
                print(f"Kept subtree: {path}")
            continue

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=False)
            removed += 1
            if verbose:
                print(f"🗑️  Deleted: {path}")
        except PermissionError as e:
            print(f"  Skipped (perm): {path} — {e}")

    if verbose:
        print(
            f"Finished pruning {root} — removed {removed}, kept {kept} protected items."
        )


def main():
    ap = argparse.ArgumentParser("dedup_corpus_simple")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--id_col", default="doc_id")
    ap.add_argument("--text_col", default="contents")

    # Input parquet to dedup:
    #  - If provided, we dedup THAT file.
    #  - Else default to out/cache/corpus_970k.parquet (Step 1 output).
    ap.add_argument("--corpus_cache", type=Path,
                    help="Path to the parquet to dedup (e.g., cache/corpus_970k.parquet or cache/queries_30k.parquet)")

    ap.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() or 8))
    ap.add_argument("--jaccard_threshold", type=float, default=0.7)
    ap.add_argument("--keep_work", action="store_true")

    # Output naming: optional; if omitted, derived from input parquet's stem.
    ap.add_argument("--file_name", default=None,
                    help="Base name for outputs (e.g., 'corpus_970k'). If omitted, uses input parquet's stem.")

    args = ap.parse_args()

    outdir = args.outdir.resolve()
    cdir = outdir / "cache"
    mdir = outdir / "manifests"
    wdir = outdir / "dedup_work"
    ensure_dir(cdir)
    ensure_dir(mdir)
    ensure_dir(wdir)

    # Choose input parquet
    corpus_parquet = resolve_input_parquet(outdir, args.corpus_cache)
    if not corpus_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {corpus_parquet}")

    # Derive output names
    base, dedup_parquet, manifest_json = derive_output_paths(outdir, corpus_parquet, args.file_name)

    print(f"• Dedup input : {corpus_parquet}")
    print(f"• Output base : {base}")
    print(f"• ID column   : {args.id_col}")
    print(f"• Text column : {args.text_col}")

    # Run dedup
    cleaned_dir = run_ray_dedup(wdir, corpus_parquet, args.id_col, args.text_col,
                                num_workers=args.num_workers, thr=args.jaccard_threshold)

    # Merge cleaned shards
    print("• Merging cleaned shards …")
    merge_cleaned_parquets(cleaned_dir, dedup_parquet)

    # Kept IDs manifest
    kept_ids = pl.read_parquet(dedup_parquet, columns=[args.id_col])[args.id_col] \
                .cast(pl.Int64).to_list()
    ensure_dir(manifest_json.parent)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump({"ids": kept_ids}, f)

    # Cleanup
    # if not args.keep_work:
    #     shutil.rmtree(wdir, ignore_errors=True)
    
    clean_folder_except(wdir,exclude="docs_to_remove_consolidated", verbose=True)

    print("✓ dedup_corpus_simple done")
    print(f"  kept rows : ~{len(kept_ids):,}")
    print(f"  parquet   : {dedup_parquet}")
    print(f"  manifest  : {manifest_json}")


if __name__ == "__main__":
    main()