import polars as pl
import os
import argparse
from pathlib import Path

def split_parquet(input_file: Path, output_dir: Path, chunk_size: int):
    """
    Reads a Parquet file and splits it into smaller chunks of a specified size.
    """
    # --- 1. Setup and Validation ---
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Read, Split, and Write Data ---
    print(f"Reading {input_file}...")
    df = pl.read_parquet(input_file)

    total_rows = df.height
    if total_rows == 0:
        print("Input file is empty. Nothing to do.")
        return
        
    # Calculate the total number of files that will be created
    num_files = (total_rows + chunk_size - 1) // chunk_size
    print(f"Total rows: {total_rows}. Splitting into {num_files} files of up to {chunk_size} rows each.")

    # Loop to create the split files
    for i in range(num_files):
        # Calculate the starting row for the slice
        offset = i * chunk_size
        
        # Slice the DataFrame to get the current chunk
        chunk = df.slice(offset, chunk_size)
        
        # Define the output filename with leading zeros for proper sorting
        output_filename = output_dir / f"part_{i + 1:04d}.parquet"
        
        if chunk.height > 0:
            print(f"Writing {chunk.height} rows to {output_filename}")
            chunk.write_parquet(output_filename)

    print("\nSplitting complete!")
# /mnt/nobackup/nbore/datapreprocessing/datav2/cc_main_500K/cc_main_500k_withid.parquet

def main():
    parser = argparse.ArgumentParser(
        description="Split a Parquet file into smaller chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
   
    parser.add_argument("--input", type=Path, default=Path("./data/test_common_crawl/part_0001.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("./data/test_common_crawl/"))
  
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=100000, 
        help="The maximum number of rows per output file."
    )
    
    args = parser.parse_args()
    
    split_parquet(args.input, args.output_dir, args.chunk_size)

if __name__ == "__main__":
    main()
