#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List
import numpy as np
import polars as pl
import shutil
import pathlib

# ---------- helpers ----------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(p: Path, obj) -> None:
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)



def init_permutations(seed: int, num_perm: int) -> np.ndarray:
    max_int = np.uint64((1 << 64) - 1)
    gen = np.random.RandomState(seed)
    permutations = np.array(
        [gen.randint(0, max_int, dtype=np.uint64) for _ in range(num_perm)],
        dtype=np.uint64,
    ).T
    permutations[permutations % 2 == 0] += 1
    return permutations



def read_parquet_columns_pl(path: Path, columns: List[str]) -> pl.DataFrame:
    return pl.read_parquet(path, columns=[c for c in columns if c])

def write_parquet_pl(path: Path, df: pl.DataFrame) -> None:
    ensure_dir(path.parent)
    df.write_parquet(path, compression="zstd")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("prepare_simple (polars)")
    ap.add_argument("--input", required=True, type=Path, help="Raw 1M parquet")
    ap.add_argument("--outdir", required=True, type=Path)

    ap.add_argument("--id_col", default="doc_id")
    ap.add_argument("--text_col", default="contents")
    ap.add_argument("--minhash_col", default=None, help="Optional: precomputed minhash column (list[uint32] length K)")

    ap.add_argument("--queries_n", type=int, default=30000)
    ap.add_argument("--global_seed", type=int, default=12345)
    ap.add_argument("--prev_index_dir", type=Path, default=None)


    # Sketch spec (frozen)
    ap.add_argument("--K", type=int, default=112)
    ap.add_argument("--M_bits", type=int, default=3584)
    ap.add_argument("--value_to_bucket", choices=["mod","mmh3"], default="mod")
    ap.add_argument("--perms_seed", type=int, default=49037)
    ap.add_argument("--mmh3_seed", type=int, default=9173)

    args = ap.parse_args()

    print("=======prev_index_dir==========")
    print(args.prev_index_dir)
    if args.prev_index_dir:
        print(args.prev_index_dir)


    outdir = args.outdir.resolve()
    mdir = outdir / "manifests"
    cdir = outdir / "cache"
    ensure_dir(mdir); ensure_dir(cdir)

    cols = [args.id_col, args.text_col]
    if args.minhash_col:
        cols.append(args.minhash_col)

    print(f"• Reading columns {cols} from {args.input} with Polars …")
    df = read_parquet_columns_pl(args.input, cols)
    if args.id_col not in df.columns or args.text_col not in df.columns:
        raise ValueError(f"Missing required columns ({args.id_col}, {args.text_col}).")
    total = df.height
    if total < args.queries_n + 1:
        raise ValueError(f"Need at least queries_n+1 rows. Have {total}, queries_n {args.queries_n}.")

    # Deterministic split
    rng = np.random.default_rng(args.global_seed)
    perm = rng.permutation(total)
    q_idx = perm[:args.queries_n]

    # ---- robust row selection for older Polars: boolean mask ----
    mask_q = np.zeros(total, dtype=bool)
    mask_q[q_idx] = True
    q_df = df.filter(pl.Series(mask_q))            # queries
    c_df = df.filter(pl.Series(~mask_q))           # corpus

    q_ids = q_df[args.id_col].cast(pl.Int64).to_list()
    c_ids = c_df[args.id_col].cast(pl.Int64).to_list()

    # Save queries parquet (small, fast)
    q_cols = [args.id_col, args.text_col] + ([args.minhash_col] if args.minhash_col else [])
    write_parquet_pl(cdir / "queries.parquet", q_df.select(q_cols))

    # NEW: Save the raw corpus parquet so Step 2 can dedup it directly
    c_cols = [args.id_col, args.text_col] + ([args.minhash_col] if args.minhash_col else [])
    write_parquet_pl(cdir / "corpus.parquet", c_df.select(c_cols))

    # Manifests
    write_json(mdir / "queries.json", {"ids": q_ids, "global_seed": args.global_seed})
    write_json(mdir / "corpus.json", {"ids": c_ids})

    write_json(mdir / "dataset_meta.json", {
        "total_rows": int(total),
        "queries_n": int(args.queries_n),
        "corpus_n": int(len(c_ids)),
        "id_col": args.id_col,
        "text_col": args.text_col,
        "minhash_col": args.minhash_col
    })

    # Freeze sketch spec + permutations (for later steps)
    spec = {
        "K": int(args.K),
        "M_bits": int(args.M_bits),
        "value_to_bucket": args.value_to_bucket,
        "perms_seed": int(args.perms_seed),
        "mmh3_seed": int(args.mmh3_seed),
        "global_seed": int(args.global_seed)
    }
    write_json(mdir / "sketch_spec.json", spec)





    perms = init_permutations(args.perms_seed, args.K)
    
    permutations_path = mdir / "permutations.npy"
  

    # np.save(mdir / "permutations.npy", perms)

    if args.prev_index_dir is not None:
        # --- Option 1: Copy from a previous index ---
        print(f"Reusing permutations from previous index: {args.prev_index_dir}")
        
        prev_perms_path = pathlib.Path(args.prev_index_dir) / "permutations.npy"
        
        if not prev_perms_path.exists():
            print(f"Error: Cannot find 'permutations.npy' in {prev_perms_path.parent}")
            return

        print(f"Copying {prev_perms_path} to {permutations_path}")
        shutil.copyfile(prev_perms_path, permutations_path)


    else:
        # --- Option 2: Create a new permutations file ---
        print("No previous index provided. Creating new permutations.")
        perms = init_permutations(args.perms_seed, args.K)
        
        print(f"Saving new permutations to {permutations_path}")
        np.save(permutations_path, perms)

    print("\nDone.")

    print("✓ prepare_simple done")
    print(f"  queries: {len(q_ids):,}  → {cdir/'queries.parquet'}")
    print(f"  corpus : {len(c_ids):,}")
    print(f"  sketch : {mdir/'sketch_spec.json'}, {mdir/'permutations.npy'}")
    print(f"  meta   : {mdir/'dataset_meta.json'}")

if __name__ == "__main__":
    main()
