#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, shutil, os
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

# ---------- Optional Ray/Spark deps ----------
try:
    from ray_local import FdedupLocal
    _HAS_RAY = True
    print(" ray_local available")
except Exception:
    _HAS_RAY = False
    print("  ray_local NOT available; this script requires it.")

# ---------- FS helpers ----------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def rm_tree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)

# ---------- Combine parquets ----------
def combine_parquets(queries_parquet: Path,
                     corpus_parquet: Path,
                     id_col: str,
                     text_col: str,
                     out_path: Path) -> Path:
    """Stream-concat two parquets, selecting only [id_col, text_col]."""
    print("queries_parquet")
    print(out_path)
    print(queries_parquet)
    print(corpus_parquet)

    ensure_dir(out_path.parent)
    # q_lf = pl.scan_parquet(queries_parquet, with_columns=[id_col, text_col])
    # c_lf = pl.scan_parquet(corpus_parquet,  with_columns=[id_col, text_col])
    q_lf = pl.scan_parquet(queries_parquet)
    c_lf = pl.scan_parquet(corpus_parquet)


    # Optional: sanity check schema
    q_cols = q_lf.collect(streaming=True).columns
    c_cols = c_lf.collect(streaming=True).columns
    if id_col not in q_cols or text_col not in q_cols:
        raise ValueError(f"{queries_parquet} missing columns {id_col}/{text_col}")
    if id_col not in c_cols or text_col not in c_cols:
        raise ValueError(f"{corpus_parquet} missing columns {id_col}/{text_col}")

    lf = pl.concat([q_lf, c_lf], how="vertical")
    lf.collect(streaming=True).write_parquet(out_path, compression="zstd")
    return out_path

# ---------- Split & Ray dedup ----------
def split_parquet(df_path: Path, out_dir: Path, num_workers: int, prefix: str = "part") -> List[Path]:
    ensure_dir(out_dir)
    ldf = pl.scan_parquet(df_path)
    n = ldf.select(pl.len()).collect(streaming=True).item()
    if n == 0:
        return []
    sz = math.ceil(n / max(1, num_workers))
    parts: List[Path] = []
    start = 0
    while start < n:
        stop = min(start + sz, n)
        chunk = ldf.slice(start, stop - start).collect(streaming=True)
        p = out_dir / f"{prefix}_{len(parts)}.parquet"
        chunk.write_parquet(p)
        parts.append(p)
        start = stop
    return parts

def run_ray_dedup(work_root: Path,
                  input_parquet: Path,
                  id_col: str,
                  text_col: str,
                  num_workers: int,
                  jaccard_threshold: float) -> Path:
    """
    Returns the path to the 'docs_to_remove_consolidated' dir (or the output dir if needed).
    """
    if not _HAS_RAY:
        raise RuntimeError("ray_local.FdedupLocal not found in env.")

    inp = work_root / "input"
    out = work_root / "output"
    ensure_dir(inp); ensure_dir(out)

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
        jaccard_similarity_threshold=jaccard_threshold,
        shingle_option="word",
        services="SignatureCalculation,ClusterAnalysis,GetDuplicateList,DataCleaning",
        operation_mode="filter_duplicates",
        run_locally=True,
    ).transform()

    # Prefer the consolidated pairs dir
    pairs_dir = out / "docs_to_remove_consolidated"
    if pairs_dir.exists():
        return pairs_dir

    # Fallback: sometimes a single file is written into output/
    cand = sorted(out.glob("*docs_to_remove_consolidated*.parquet"))
    if cand:
        return cand[0]

    raise FileNotFoundError(f"Could not find docs_to_remove_consolidated under {out}")

# ---------- Build GT from dedup pairs ----------
def read_queries_ids(queries_parquet: Path, id_col: str) -> List[int]:
    # lf = pl.scan_parquet(queries_parquet, with_columns=[id_col])
    lf = pl.scan_parquet(queries_parquet)
    return lf.collect(streaming=True).get_column(id_col).cast(pl.Int64).to_list()

def _read_pairs_any(source: Path) -> pl.DataFrame:
    """Read docs_to_remove_consolidated as a single DataFrame with [first_doc, docs_to_remove]."""
    if source.is_dir():
        parts = sorted(source.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet files in directory: {source}")
        lf = pl.concat([pl.scan_parquet(p) for p in parts], how="vertical")
    else:
        lf = pl.scan_parquet(source)
    df = (
        lf.select(
            pl.col("first_doc").cast(pl.Int64),
            pl.col("docs_to_remove").cast(pl.Int64),
            pl.col("docs_to_remove_distance"),
        )
        .unique()
        .collect(streaming=True)
    )
    return df

def build_gt_for_queries(queries: List[int], pairs_df: pl.DataFrame, gt_k: int, include_empty: bool) -> Dict[int, List[int]]:
    # canonical -> members
    canon_tbl = (
        pairs_df
        .group_by("first_doc")
        .agg(pl.col("docs_to_remove"))
        .rename({"first_doc": "canon", "docs_to_remove": "members"})
    )
    canon_to_members: Dict[int, List[int]] = {
        int(row["canon"]): [int(x) for x in row["members"]]
        for row in canon_tbl.iter_rows(named=True)
    }
    # member -> canonical
    mem_to_canon: Dict[int, int] = {
        int(r["docs_to_remove"]): int(r["first_doc"])
        for r in pairs_df.iter_rows(named=True)
    }

    qset = set(int(q) for q in queries)
    gt: Dict[int, List[int]] = {}

    for qid in qset:
        nbrs = set()

        # If qid is canonical, neighbors are all its members
        mems = canon_to_members.get(qid)
        if mems:
            nbrs.update(mems)

        # If qid is member, neighbors are canonical + other members (except itself)
        canon = mem_to_canon.get(qid)
        if canon is not None:
            nbrs.add(canon)
            for m in canon_to_members.get(canon, []):
                if m != qid:
                    nbrs.add(m)

        nbrs.discard(qid)
        if nbrs or include_empty:
            lst = sorted(nbrs)
            if gt_k > 0:
                lst = lst[:gt_k]
            gt[qid] = lst

    return gt


from typing import Dict, List, Tuple
import polars as pl

def build_gt_for_queries_with_distances(
    queries: List[int],
    pairs_df: pl.DataFrame,
    gt_k: int,
    include_empty: bool,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Ground truth with distances:
      gt_dist[qid] = [(neighbor_id, distance), ...]
    """

    # canonical -> list[(member_id, dist)]
    canon_to_members: Dict[int, List[Tuple[int, float]]] = {}
    for row in pairs_df.iter_rows(named=True):
        print(row)
        canon = int(row["first_doc"])
        mem   = int(row["docs_to_remove"])
        dlist = row["docs_to_remove_distance"]  # Python list[float]
        dist  = float(dlist[0]) if dlist else 0.0

        canon_to_members.setdefault(canon, []).append((mem, dist))

    # member -> (canonical, dist)
    mem_to_canon: Dict[int, Tuple[int, float]] = {}
    for row in pairs_df.iter_rows(named=True):
        canon = int(row["first_doc"])
        mem   = int(row["docs_to_remove"])
        dlist = row["docs_to_remove_distance"]
        dist  = float(dlist[0]) if dlist else 0.0
        # if duplicates exist, keep the smallest distance
        prev = mem_to_canon.get(mem)
        if prev is None or dist < prev[1]:
            mem_to_canon[mem] = (canon, dist)

    qset = set(int(q) for q in queries)
    gt_dist: Dict[int, List[Tuple[int, float]]] = {}

    for qid in qset:
        # use dict so we can keep the *best* distance if a neighbor appears twice
        nbrs: Dict[int, float] = {}

        # Case 1: qid is canonical → neighbors are its members (with dist)
        for mem, dist in canon_to_members.get(qid, []):
            if mem == qid:
                continue
            prev = nbrs.get(mem)
            if prev is None or dist < prev:
                nbrs[mem] = dist

        # Case 2: qid is a member → neighbor is canonical (with dist)
        mc = mem_to_canon.get(qid)
        if mc is not None:
            canon, dist_c = mc
            if canon != qid:
                prev = nbrs.get(canon)
                if prev is None or dist_c < prev:
                    nbrs[canon] = dist_c

            # and other members of that canonical cluster
            for mem, dist_m in canon_to_members.get(canon, []):
                if mem == qid:
                    continue
                # we only know member→canonical distances; use the member’s
                # own distance to canonical as an approximate edge weight.
                approx = dist_m
                prev = nbrs.get(mem)
                if prev is None or approx < prev:
                    nbrs[mem] = approx

        # remove self if present
        nbrs.pop(qid, None)

        if nbrs or include_empty:
            items = sorted(nbrs.items(), key=lambda t: t[1])  # sort by distance
            if gt_k > 0:
                items = items[:gt_k]
            gt_dist[qid] = items

    return gt_dist


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser("build_gt_joint_step5")
    # I/O
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--queries_parquet", required=True, type=Path)
    ap.add_argument("--corpus_parquet",  required=True, type=Path)
    ap.add_argument("--corpus_raw_parquet",  required=False, type=Path)
    ap.add_argument("--id_col", default="doc_id")
    ap.add_argument("--text_col", default="contents")

    # Dedup params
    ap.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() or 8))
    ap.add_argument("--jaccard_threshold", type=float, default=0.7)

    # GT params
    ap.add_argument("--gt_k", type=int, default=1000)
    ap.add_argument("--include_empty", action="store_true")
    ap.add_argument("--out_json", type=Path, default=None)

    ap.add_argument("--prev_index_dir", type=Path, default=None)






    # keep thigns
    ap.add_argument("--keep_work", action="store_true")
    ap.add_argument("--keep_combined", action="store_true")
    ap.add_argument("--combined_name", default=None,
                    help="Optional file base for combined parquet (else auto from stems)")

    args = ap.parse_args()

    print("=======prev_index_dir==========")
    print(args.prev_index_dir)


    outdir = args.outdir.resolve()
    work = outdir / "dedup_work_gt"
    cache = outdir / "cache"
    gt_dir = outdir / "ground_truth"
    ensure_dir(work); ensure_dir(cache); ensure_dir(gt_dir)


    #  >>>>
    if args.prev_index_dir:
        # 1) Combine parquets. with prev
        c_stem = args.corpus_parquet.stem
        base = f"__PLUS_PREV__{c_stem}"
        prev_file=f"{args.prev_index_dir}/corpus_970k_dedup.parquet"

        prev_combined_parquet = cache / "corpus_970k_dedup.parquet"

        print(f"• Combining:\n  - {prev_file}\n  - {args.corpus_parquet}\n→ {prev_combined_parquet}")
        combine_parquets(prev_file, args.corpus_parquet, args.id_col, args.text_col, prev_combined_parquet)

        print("SKIPPNG")
        # # merge raw file for milvus

        # prev_file=f"{args.prev_index_dir}/corpus_970k.parquet"

        # prev_combined_parquet = cache / "corpus_970k.parquet"

        # print(f"• Combining:\n  - {prev_file}\n  - {args.corpus_parquet}\n→ {prev_combined_parquet}")
        # combine_parquets(prev_file, args.corpus_raw_parquet, args.id_col, args.text_col, prev_combined_parquet)

    else:
        print("=======NOT  ADDED==========")
     #  >>>>

    # 1) Combine parquets
    q_stem = args.queries_parquet.stem
    c_stem = args.corpus_parquet.stem
    base = args.combined_name or f"{q_stem}__PLUS__{c_stem}"
    base = f"{q_stem}__PLUS__{c_stem}"

    combined_parquet = cache / f"{base}.parquet"
    #  >>>>
    print(f"• Combining:\n  - {args.queries_parquet}\n  - {args.corpus_parquet}\n→ {combined_parquet}")
    combine_parquets(args.queries_parquet, args.corpus_parquet, args.id_col, args.text_col, combined_parquet)
    #  >>>>
    # # we now combine with prev
    # print(f"• Combining:\n  - {args.queries_parquet}\n  - {prev_combined_parquet}\n→ {combined_parquet}")
    # combine_parquets(args.queries_parquet, prev_combined_parquet, args.id_col, args.text_col, combined_parquet)


    # 2) Run dedup on the combined file
    print("• Running Ray dedup on combined data …")
    pairs_path = run_ray_dedup(work, combined_parquet, args.id_col, args.text_col,
                               num_workers=args.num_workers,
                               jaccard_threshold=args.jaccard_threshold)
    print(f"• Dedup pairs located at: {pairs_path}")

    # 3) Build GT for queries
    print("• Building GT from docs_to_remove_consolidated …")
    queries = read_queries_ids(args.queries_parquet, args.id_col)
    pairs_df = _read_pairs_any(pairs_path)
    gt = build_gt_for_queries(queries, pairs_df, gt_k=max(0, args.gt_k), include_empty=args.include_empty)


    gt_dist = build_gt_for_queries_with_distances(
    queries, pairs_df, gt_k=max(0, args.gt_k), include_empty=args.include_empty
    )
    out_json = (gt_dir / f"gt_top{args.gt_k}_with_distances.json")
    ensure_dir(out_json.parent)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                str(qid): [
                    {"id": int(nid), "distance": float(d)}
                    for (nid, d) in neighs
                ]
                for qid, neighs in gt_dist.items()
            },
            f,
            indent=2,
        )


    out_json = args.out_json or (gt_dir / f"gt_top{args.gt_k}.json")
    ensure_dir(out_json.parent)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in gt.items()}, f, indent=2)

    n_with = sum(1 for v in gt.values() if len(v) > 0)
    print("✓ GT built")
    print(f"  queries total   : {len(queries):,}")
    print(f"  queries with GT : {n_with:,}")
    print(f"  → {out_json}")

    # 4) Cleanup
    if not args.keep_work:
        rm_tree(work)
    if not args.keep_combined:
        try:
            combined_parquet.unlink(missing_ok=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()
