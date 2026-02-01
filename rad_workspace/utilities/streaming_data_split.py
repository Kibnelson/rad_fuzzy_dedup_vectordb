#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl


def split_parquet_streaming(input_file: Path, output_dir: Path, chunk_size: int) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan (lazy) – does NOT load whole file
    ldf = pl.scan_parquet(str(input_file))

    # Get row count (can still be expensive, but memory-safe)
    total_rows = ldf.select(pl.len()).collect(streaming=True).item()
    if total_rows == 0:
        print("Input file is empty. Nothing to do.")
        return

    num_files = (total_rows + chunk_size - 1) // chunk_size
    print(f"Total rows: {total_rows}. Splitting into {num_files} files of up to {chunk_size} rows each.")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")

    part = 1
    for offset in range(0, total_rows, chunk_size):
        out_path = output_dir / f"part_{part:04d}.parquet"

        # Slice lazily, then collect with streaming => only this chunk is materialized
        df_chunk = ldf.slice(offset, chunk_size).collect(streaming=True)

        if df_chunk.height == 0:
            break

        print(f"Writing rows [{offset}:{offset + df_chunk.height}) -> {out_path}")
        df_chunk.write_parquet(out_path)
        part += 1

    print("Splitting complete!")


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--input", type=Path, required=True)
    # ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--input", type=Path, default=Path("./data/100M.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("./data/cc_main_rad_100M_NEW/"))


    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    args = ap.parse_args()

    split_parquet_streaming(args.input, args.output_dir, args.chunk_size)


if __name__ == "__main__":
    main()
