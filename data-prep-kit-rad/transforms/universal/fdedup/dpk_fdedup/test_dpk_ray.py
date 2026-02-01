
from ray_local import Fdedup
# from local_transform.local_tranform import Fdedup
import time
from typing import List, Tuple, Dict, Any, Iterable
from pathlib import Path
import polars as pl
import math
import os
import math
import argparse
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed


start_time_dedup = time.perf_counter_ns()


#   # Single argument for service execution
#     parser.add_argument(
#         "--services",
#         type=str,
#         required=False,
#         default="SignatureCalculation,ClusterAnalysis,GetDuplicateList,DataCleaning",
#         help="Comma-separated list of services to run (e.g., SignatureCalculation,ClusterAnalysis,GetDuplicateList,DataCleaning)",
#     )

    # services= "SignatureCalculation,ClusterAnalysis,GetDuplicateList",


# input_folder="/mnt/nobackup/nbore/datapreprocessing/data-prep-kit2/data-prep-kit/transforms/universal/fdedup/python/src/exp/input_data/0"
# output_folder="/mnt/nobackup/nbore/datapreprocessing/data-prep-kit2/data-prep-kit/transforms/universal/fdedup/python/src/exp/output_data/0"

# Fdedup(input_folder='/mnt/nobackup/nbore/datapreprocessing/data-prep-kit2/data-prep-kit/transforms/universal/fdedup/python/test-data/c4/input/',
#     output_folder='/mnt/nobackup/nbore/datapreprocessing/data-prep-kit2/data-prep-kit/transforms/universal/fdedup/python/test-data/c4/output/',
#     contents_column= "contents",
#     document_id_column= "int_id_column",
#     num_permutations= 112,
#     num_bands= 14,
#     num_minhashes_per_band= 8,
#     jaccard_similarity_threshold= 0.7,
#     shingle_option="word",
#     operation_mode="filter_duplicates",
#     run_locally= True).transform()

# /mnt/nobackup/nbore/datapreprocessing/datav2/c4/raw_file


# Fdedup(input_folder='/mnt/nobackup/nbore/datapreprocessing/data-prep-kit2/data-prep-kit/transforms/universal/fdedup/python/src/exp/input_data/0/',
#     output_folder='/mnt/nobackup/nbore/datapreprocessing/data-prep-kit2/data-prep-kit/transforms/universal/fdedup/python/src/exp/output_data/0/',
#     contents_column= "contents",
#     document_id_column= "int_id_column",
#     num_permutations= 112,
#     num_bands= 14,
#     num_minhashes_per_band= 8,
#     jaccard_similarity_threshold= 0.7,
#     shingle_option="word",
#     operation_mode="filter_duplicates",
#     run_locally= True).transform()

def split_parquet_dpk(input_file: str,
                  num_workers: int,
                  output_dir: str,
                  prefix: str) -> None:
    # 1) Read the full parquet file once
    df = pl.read_parquet(input_file)
    total = df.height
    if total == 0:
        print("Input file has no rows. Exiting.")
        return
        
    # 2) Compute chunk boundaries
    chunk_size = math.ceil(total / num_workers)
    os.makedirs(output_dir, exist_ok=True)

    # 3) Worker that writes one slice
    def write_shard(i):
        start = i * chunk_size
        if start >= total:
            return i, 0, None
        end = min((i + 1) * chunk_size, total)
        chunk = df[start:end]                 # Polars slice is zero-copy view
        out_path = os.path.join(output_dir, f"{prefix}_{i}.parquet")
        chunk.write_parquet(out_path)
        return i, chunk.height, out_path

    # 4) Fire up threads
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(write_shard, i) for i in range(num_workers)]
        for future in as_completed(futures):
            i, nrows, path = future.result()
            if path:
                print(f"[worker {i:02d}] wrote {nrows} rows → {path}")


final_input_output="/mnt/nobackup/nbore/datapreprocessing/datav2/lm1b/raw_file/output_test/"
final_input_output="/mnt/nobackup/nbore/datapreprocessing/data-prep-kit2/data-prep-kit/transforms/universal/fdedup/python/test-data/c4/"

final_input_local=final_input_output+"/input"
final_output_local=final_input_output+"/output"

# output_file = "/mnt/nobackup/nbore/datapreprocessing/datav2/lm1b/raw_file/lm1b_combined_3m_shuffled.parquet"
output_file = "/mnt/nobackup/nbore/datapreprocessing/data-prep-kit2/data-prep-kit/transforms/universal/fdedup/python/test-data/c4/c4_30m_withid_shuffled.parquet"

num_of_threads=34

split_parquet_dpk(output_file,num_of_threads,final_input_local,"part")



Fdedup(input_folder=final_input_local,
    output_folder=final_output_local,
    contents_column= "contents",
    document_id_column= "int_id_column",
    num_permutations= 112,
    num_bands= 14,
    num_minhashes_per_band= 8,
    jaccard_similarity_threshold= 0.7,
    shingle_option="word",
    operation_mode="filter_duplicates",
    run_locally= True).transform()

start_time_dedup = time.perf_counter_ns()

end_time_dedup = time.perf_counter_ns()
dedup_time_ns = (end_time_dedup - start_time_dedup) / 1e9
print(f"Chunk processing dedup_time_ns >>> : {dedup_time_ns:.2f} sec")
