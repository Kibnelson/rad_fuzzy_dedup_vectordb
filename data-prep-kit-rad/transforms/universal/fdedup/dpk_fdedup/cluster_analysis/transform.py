# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM Corp. 2024.
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import io
import os
import re
from argparse import ArgumentParser, Namespace
from typing import Any, List

import numpy as np
import polars as pl
from data_processing.transform import AbstractFolderTransform, TransformConfiguration
from data_processing.utils import (
    CLIArgumentProvider,
    TransformUtils,
    UnrecoverableException,
    get_logger,
)
from dpk_fdedup.Murmur_MH import Murmur_MH
import time
from pathlib import Path
import argparse, os, random, shutil, sys, json


short_name = "cluster"
cli_prefix = f"{short_name}_"

# configuration keys
num_bands_key = "num_bands"
""" This key holds the number of bands used in the banding technique"""
num_segments_key = "num_segments"
""" This key holds the number of segments dividing the hashing space for each band"""
jaccard_similarity_threshold_key = "jaccard_similarity_threshold"
""" This key holds the Jaccard similarity threshold above which two documents are duplicates"""
sort_output_key = "sort_output"
""" This key is used to sort"""

# command line arguments
num_bands_cli_param = f"{cli_prefix}{num_bands_key}"
""" The number of bands used in the banding technique"""
jaccard_similarity_threshold_cli_param = f"{cli_prefix}{jaccard_similarity_threshold_key}"
""" Jaccard similarity threshold above which two documents are duplicates"""
num_segments_cli_param = f"{cli_prefix}{num_segments_key}"
""" The number of segments dividing the hashing space for each band"""
sort_output_cli_param = f"{cli_prefix}{sort_output_key}"
""" Sort the output"""

captured_arg_keys = [
    num_bands_key,
    num_segments_key,
    jaccard_similarity_threshold_key,
    sort_output_key,
]

# defaults
num_bands_default = 14
""" Default number of bands used in the banding technique (from FineWeb https://arxiv.org/pdf/2406.17557)"""
jaccard_similarity_threshold_default = 0.7
""" Default Jaccard similarity threshold (from FineWeb https://arxiv.org/pdf/2406.17557)"""
num_segments_default = 1
""" Default number of segments dividing the hashing space for each band"""
sort_output_default = False


class ClusterAnalysisTransform(AbstractFolderTransform):
    """
    This is the second transform of the fuzzy dedup pipeline. It runs in parallel:
    for each band, the hashing interval is divided into segments. A cluster analysis
    uses as input all the parquet files from segment of a band. The `bands` output
    of the signature calculation, the first transform in the fuzzy dedup pipeline
    contains all the data for a given segment s of a specific band b in the
    subfolder `bands/band=b/segment=s`.
    The transform loads all the parquet files in the `bands/band=b/segment=s`
    subfolder. Each one of these parquet files has two columns: the `band_hash`
    and a `data` structure, which includes the `document_id`, the `minhashes` and
    the `document_size` fields. Once all the files have been loaded in a single
    dataframe, a `group_by` operation on the `band_hash` field is performed in
    that dataframe. All the documents that have the same band_hash are grouped
    in a cluster. Subsequently, the documents of each cluster are sorted in
    descending order according to their size, and a Jaccard similarity is
    calculated between the cluster documents. The documents for which the Jaccard
    similarity is above the `jaccard_similarity_threshold` remain in the cluster,
    the others are removed from the cluster. Finally, from each cluster that has
    more than one document after running the Jaccard similarity, we select a doc
    to keep (the largest size document), and mark the other documents as
    duplicates. The resulting clusters are saved in a file for further analysis.

    The following internal variables are initialized from the config parameter:
        num_bands: number of bands used in the banding technique
        jaccard_similarity_threshold: Jaccard similarity threshold above which two documents are duplicates
        num_segments: the number of segments dividing the hashing space for each band
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize based on the dictionary of configuration information.
        This is generally called with configuration parsed from the CLI arguments
        defined by the companion runtime, ClusterAnalysisTransformRuntime.
        """
        super().__init__(config)
        self.num_bands = config.get(num_bands_key, num_bands_default)
        self.num_segments = config.get(num_segments_key, num_segments_default)
        self.jaccard_similarity_threshold = config.get(
            jaccard_similarity_threshold_key, jaccard_similarity_threshold_default
        )
        self.sort_output = config.get(sort_output_key, sort_output_default)
        self.data_access = config.get("data_access")
        if self.data_access is None:
            raise UnrecoverableException("Could not get a pointer to the data access object inside the transform.")
        self.logger = get_logger(__name__)


    # def get_base_output_dir(self,path: str) -> str:
    #     """
    #     Given a full file path, return only up to the 'output/' directory.
    #     Example:
    #         /mnt/.../output/docs_to_remove/band_13_segment_0.parquet
    #     -> 
    #         /mnt/.../output/
    #     """
    #     p = Path(path).resolve()
    #     # Walk up until we find "output"
    #     for parent in p.parents:
    #         if parent.name == "output":
    #             return str(parent) + "/"   # add trailing slash for clarity
    #     return str(p.parent) + "/"        # fallback: just parent dir

    def base_output_append(self, path: str, append_path: str) -> str:
        """
        Given a file path, return the path up to 'output/' and append the user-specified subpath.

        Example:
            path = "/mnt/.../output/docs_to_remove/band_13_segment_0.parquet"
            append_path = "bands/text/"
        -> 
            "/mnt/.../output/bands/text/"
        """
        p = Path(path).resolve()
        for parent in p.parents:
            if parent.name == "output":
                return str(parent / append_path)
        raise ValueError("No 'output' directory found in path")


    def remove_folder_and_contents(self, folder_path: str | Path, *, verbose: bool = True) -> None:
        """
        Delete all files (and sub‑folders) in `folder_path` and then delete the
        folder itself.

        Parameters
        ----------
        folder_path : str | pathlib.Path
            Path to the directory you want to remove.
        verbose : bool, default True
            Print a confirmation message.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        NotADirectoryError
            If the path exists but is not a directory.
        PermissionError
            If permissions prevent deletion.
        """
        p = Path(folder_path).expanduser().resolve()
        if not p.exists():
            print(f"🗑️  Does no exist: {p}")
        else:
            shutil.rmtree(p)
            if verbose:
                print(f"🗑️  Deleted directory and all contents: {p}")

    def transform(self, folder_name: str) -> tuple[list[tuple[bytes, str]], dict[str, Any]]:
        # self.logger.debug(f"Cluster analysis for folder {folder_name}")
        # metadata = {}
        # input_folder = TransformUtils.clean_path(os.path.join(self.data_access.input_folder, folder_name))
        # files, retries = self.data_access.get_folder_files(
        #     path=input_folder,
        #     extensions=[".parquet"],
        #     return_data=True,
        # )
        # if retries > 0:
        #     metadata |= {"data_access_retries": retries}
        # match = re.match(r"^band=(\d+)/segment=(\d+)$", folder_name)
        # if match:
        #     band = int(match.group(1))
        #     segment = int(match.group(2))
        # else:
        #     match = re.match(r"^band=(\d+)\\segment=(\d+)$", folder_name)
        #     if match:
        #         band = int(match.group(1))
        #         segment = int(match.group(2))
        #     else:
        #         raise ValueError(
        #             f"Wrong folder_name {folder_name}, should be either band=b/segment=s or band=b\\segment=s (windows)"
        #         )
        # output_folder = TransformUtils.clean_path(self.data_access.output_folder)
        # output_path = os.path.join(output_folder, f"band_{band}_segment_{segment}.parquet")

        # # consolidate into a single data frame band hashes computed by workers
        # band_segment_dataframe, consolidation_stats = self._consolidate_band_segment_files(files)
        # metadata |= consolidation_stats
        # # cluster grouping by band hashes
        # cluster_dataframe, cluster_stats = self._get_clusters(band_segment_dataframe)
        # metadata |= cluster_stats
        # # cluster analysis using jaccard similarity
        # jaccard_cluster_dataframe, jaccard_stats = self._analyze_clusters(cluster_dataframe)
        # metadata |= jaccard_stats
        # # Generate the docs_to_remove dataframe
        # docs_to_remove_dataframe = jaccard_cluster_dataframe.explode("docs_to_remove")
        # output_data = TransformUtils.convert_arrow_to_binary(docs_to_remove_dataframe.to_arrow())
        # self.logger.debug(f"{len(docs_to_remove_dataframe)} documents marked to remove")
        # metadata |= {"num_duplicate_documents": len(docs_to_remove_dataframe)}
        # return [(output_data, output_path)], metadata

        self.logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>Cluster analysis for folder {folder_name}")

        # path="/Users/nelson/workspace/Research/DataPreprocessing/ibm/active/data-prep-kit/transforms/universal/fdedup/python/output/data_test/"+folder_name
        #
        # # Create the directory
        # os.makedirs(path, exist_ok=True)
        # print(f"Directory created at: {path}")

        inverteFileIndex2= {}
        # data = pl.read_parquet("/Users/nelson/workspace/Research/DataPreprocessing/ibm/active/data-prep-kit/transforms/universal/fdedup/python/src/indexes/index.parquet")
        data = pl.DataFrame()

        metadata = {}
        input_folder = self.sanitize_folder_name(os.path.join(self.data_access.input_folder, folder_name))
        files, retries = self.data_access.get_folder_files(
            path=input_folder,
            extensions=[".parquet"],
            return_data=True,
        )
        if retries > 0:
            metadata |= {"data_access_retries": retries}
        match = re.match(r"^band=(\d+)/segment=(\d+)$", folder_name)
        if match:
            band = int(match.group(1))
            segment = int(match.group(2))
        else:
            raise ValueError(f"Wrong folder_name {folder_name}, should be band=b/segment=s")
        output_folder = self.sanitize_folder_name(self.data_access.output_folder)
        output_path = os.path.join(output_folder, f"band_{band}_segment_{segment}.parquet")


        t_start = time.perf_counter_ns()
      
        # consolidate into a single data frame band hashes computed by workers
        # print(files)
        band_segment_dataframe, consolidation_stats = self.consolidate_band_segment_files(files)
        metadata |= consolidation_stats
        # cluster grouping by band hashes
        # print("cluster grouping by band hashes")
        # print(band_segment_dataframe)
        # print(band_segment_dataframe.shape)
        # print(consolidation_stats)
        # band_segment_dataframe.write_parquet("/Users/nelson/workspace/Research/DataPreprocessing/ibm/active/data-prep-kit/transforms/universal/fdedup/python/src/band_segment_dataframe.parquet")

        cluster_dataframe, cluster_stats = self.get_clusters(band_segment_dataframe)

        # cluster_dataframe.write_parquet(f"/Users/nelson/workspace/Research/DataPreprocessing/ibm/active/data-prep-kit/transforms/universal/fdedup/python/src/band_{band}_segment_{segment}_cluster_dataframe.parquet")



        # # Generate a timestamped filename
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # filename = f"data_{timestamp}.parquet"
        # full_path = os.path.join(path, filename)
        # print(f"Writing {full_path}")
        # cluster_dataframe.write_parquet(full_path)
        data=[]
        metadata |= cluster_stats
        # cluster analysis using jaccard similarity
        jaccard_cluster_dataframe, jaccard_stats = self.analyze_clusters(cluster_dataframe,inverteFileIndex2,data,band)



        # print(">>>>>>>>>>>right>>>>>>>>>>>>>>>>>")
        # print(right)
        # print(len(right))
        # print(len(set(right)))
        # self.append_values_to_json(list(set(data)), f'output{band}.json')


        metadata |= jaccard_stats
        # Generate the docs_to_remove dataframe
        docs_to_remove_dataframe = jaccard_cluster_dataframe.explode("docs_to_remove")
        output_data = TransformUtils.convert_arrow_to_binary(docs_to_remove_dataframe.to_arrow())
        self.logger.info(f"{len(docs_to_remove_dataframe)} documents marked to remove")
        metadata |= {"num_duplicate_documents": len(docs_to_remove_dataframe)}
        # inverteFileIndex2.save_index()

        t_end = time.perf_counter_ns()
        # Compute and print elapsed nanoseconds
        elapsed_ns = t_end - t_start
        print(f"Elapsed time>>>>>>>>>>>>>>>>>>>CANDIDATE VERIFICATION>>>>>>>>>>>>>>>>>>>>{output_path}>>>>>>>>>: {elapsed_ns} ns")
        self.logger.info(f"Elapsed time>>>>>>>>>>>>>>>>>>>CANDIDATE VERIFICATION>>>>>>>>>>>>>>>{output_path}>>>>>>>>>>>>>>: {elapsed_ns} ns")

        # # Clean up
        # path=self.base_output_append(output_path,"bands/")
        # print(f"Elapsed time>>>>>>>>>>>>>>>>>>>NEW PATH>>>>>>>>>>>>>>>>>>>{output_path}>>>>>>>>>: {elapsed_ns} ns")
        # self.remove_folder_and_contents(path)

        return [(output_data, output_path)], metadata

    def _consolidate_band_segment_files(self, files: dict[str, bytes]) -> tuple[pl.DataFrame, dict[str, Any]]:
        band_segment_dataframe = pl.DataFrame()
        total_input_rows = 0
        for fname, contents in files.items():
            df = pl.read_parquet(io.BytesIO(contents))
            total_input_rows += len(df)
            self.logger.debug(f"{fname} has {len(df)} rows")
            band_segment_dataframe = band_segment_dataframe.vstack(df)

        consolidation_stats = {
            "input_files": len(files),
            "input_bytes": sum(len(v) for v in files.values()),
            "input_rows": total_input_rows,
            "consolidated_files": 1,
            "consolidated_bytes": band_segment_dataframe.to_arrow().nbytes,
            "consolidated_rows": len(band_segment_dataframe),
        }
        return band_segment_dataframe, consolidation_stats

    def _get_clusters(self, band_segment_dataframe: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
        groupby_dataframe = band_segment_dataframe.group_by("band_hash").agg("document_data")
        cluster_dataframe = groupby_dataframe.with_columns(cluster_length=pl.col("document_data").list.len()).filter(
            pl.col("cluster_length") > 1
        )
        # self.logger.info(f"file_name = {file_name}")
        num_clusters = len(cluster_dataframe)
        if num_clusters > 0:
            sum_cdocs = cluster_dataframe.select(pl.sum("cluster_length")).item()
            max_cdocs = cluster_dataframe.select(pl.max("cluster_length")).item()
            min_cdocs = cluster_dataframe.select(pl.min("cluster_length")).item()
            avg_cdocs = cluster_dataframe.select(pl.mean("cluster_length")).item()
        else:
            sum_cdocs = 0
            max_cdocs = 0
            min_cdocs = 0
            avg_cdocs = 0
        self.logger.debug(f"After GroupBy: {num_clusters} clusters with {sum_cdocs} total docs")
        self.logger.debug(f" max/min/avg docs per cluster: {max_cdocs}/{min_cdocs}/{avg_cdocs:.2f}")
        cluster_stats = {
            "groupby_clusters": num_clusters,
            "cluster_duplicate_docs": sum_cdocs,
        }
        return cluster_dataframe, cluster_stats

    def _analyze_clusters(self, df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
        # Define the schema with specific data types
        schema = {"first_doc": pl.Int64, "docs_to_remove": pl.List(pl.Int64),"docs_to_remove_distance": pl.List(pl.Float64), "docs_to_remove_length": pl.Int64}
        doc_ids_lists = []
        docs_to_remove_lists = []
        docs_to_remove_distance_lists = []

        len_of_docs2remove_lists = []
        for row in df.iter_rows(named=True):
            doc_ids_list, docs_to_remove_list, len_of_docs2remove_list,docs_to_remove_distance_list = self._jaccard_distance_calculation(row)
            doc_ids_lists += doc_ids_list
            docs_to_remove_lists += docs_to_remove_list
            docs_to_remove_distance_lists += docs_to_remove_distance_list

            len_of_docs2remove_lists += len_of_docs2remove_list
        jaccard_cluster_dataframe = pl.DataFrame(
            {
                "first_doc": doc_ids_lists,
                "docs_to_remove": docs_to_remove_lists,
                "docs_to_remove_distance": docs_to_remove_distance_lists,
                "docs_to_remove_length": len_of_docs2remove_lists,
            },
            schema=schema,
        )
        filtered_jaccard_dataframe = jaccard_cluster_dataframe.filter(pl.col("docs_to_remove_length") > 0)
        num_clusters = len(filtered_jaccard_dataframe)
        if num_clusters > 0:
            sum_cdocs = filtered_jaccard_dataframe.select(pl.sum("docs_to_remove_length")).item()
            max_cdocs = filtered_jaccard_dataframe.select(pl.max("docs_to_remove_length")).item()
            min_cdocs = filtered_jaccard_dataframe.select(pl.min("docs_to_remove_length")).item()
            avg_cdocs = filtered_jaccard_dataframe.select(pl.mean("docs_to_remove_length")).item()
        else:
            sum_cdocs = 0
            max_cdocs = 0
            min_cdocs = 0
            avg_cdocs = 0
        self.logger.debug(f"After Jaccard: {num_clusters} clusters with {sum_cdocs} total docs")
        self.logger.debug(f" max/min/avg docs per cluster: {max_cdocs}/{min_cdocs}/{avg_cdocs:.2f}")
        jaccard_stats = {
            "jaccard_clusters": num_clusters,
            "jaccard_duplicate_docs": sum_cdocs,
        }
        if self.sort_output:
            filtered_jaccard_dataframe = filtered_jaccard_dataframe.sort(by="first_doc")
        return filtered_jaccard_dataframe, jaccard_stats

    # def _jaccard_distance_calculation_1(self, row: List[pl.Series]) -> list[list]:
    def jaccard_distance_calculation(self, row: List[pl.Series], inverteFileIndex2, data,band) -> list[list]:
        # Process row and return a new list of Series or a new row
        threshold = self.jaccard_similarity_threshold
        doc_ids_list = []
        docs_to_remove_list = []
        docs_to_remove_distance_list = []

        len_of_docs2remove_list = []
        # sort documents
        document_data = row["document_data"]

        # Sort the list by 'document_length'
        sorted_document_data = sorted(document_data, key=lambda x: (x["int_id_column"]))

        # sorted_document_data = sorted(document_data,
        #                       key=lambda x: x["int_id_column"],
        #                       reverse=True)
        # or in-place:
        # document_data.sort(key=lambda x: x["int_id_column"], reverse=True)

        
        # sorted_document_data = sorted(document_data, key=lambda x: (-x["document_length"], x["int_id_column"]))
        # sorted_document_data = sorted(document_data, key=lambda x: (-x["document_length"], x["int_id_column"]))

        # Extracting int_id_column values into a list
        doc_list = [item["int_id_column"] for item in sorted_document_data]
        total_doc_list = len(doc_list)
        # Creating a dictionary with int_id_column as key and minhashes as value
        doc_minhashes = {item["int_id_column"]: item["minhashes"] for item in sorted_document_data}

        keys = list(doc_minhashes.keys())
        num_items = len(keys)



        # # Nested loops to iterate over each unique pair of items
        # for i in range(num_items):
        #     key_i = keys[i]
        #     # Directly access the array corresponding to key_i
        #     array1 = doc_minhashes[key_i]
        #
        #     for j in range(i + 1, num_items):
        #         key_j = keys[j]
        #         # Directly access the array corresponding to key_j
        #         array2 = doc_minhashes[key_j]
        #         #
        #         # # Now you can compare array1 and array2 or perform other operations
        #         # print(f"Comparing arrays for keys {key_i} and {key_j}")
        #         # # Example operation: print the arrays
        #         # print("Array1:", array1)
        #         # print("Array2:", array2)
        #
        #         distance_index = Murmur_MH.jaccard(np.array(array1), np.array(array2))
        #         if distance_index >= threshold:
        #
        #             data.append(key_i)
        #

        # minhashes = doc_minhashes  # adjust if needed
        #
        # for key_i, key_j in itertools.combinations(doc_minhashes.keys(), 2):
        #     array1 = minhashes[doc_minhashes[key_i]] if isinstance(minhashes, dict) else doc_minhashes[key_i]
        #     array2 = minhashes[doc_minhashes[key_j]] if isinstance(minhashes, dict) else doc_minhashes[key_j]
        #
        #     print(f"Comparing arrays for keys {key_i} and {key_j}")


        # if len(doc_list)> 20:
        #     self.add(doc_minhashes)
        #     print(doc_minhashes)
        #     print(len(doc_list))
        # docs_to_remove = []
        # new_doc_list = []
        # for (key1, key2) in combinations(doc_minhashes.keys(), 2):  # Generate all combinations of keys
        #     query_hashes = doc_minhashes[key1]
        #     doc_id_target = key2
        #
        #     total_count_ratio = inverteFileIndex2.get_count(key1, doc_id_target)
        #     if total_count_ratio == 0:
        #         total_sum = self.index_query_comparison2(query_hashes, doc_id_target, data)
        #         total_count_ratio = total_sum / len(query_hashes)
        #         inverteFileIndex2.insert_doc_id(key1, doc_id_target, total_count_ratio)
        #
        #     if total_count_ratio >= threshold:
        #         docs_to_remove.append(doc_id_target)
        # if len(docs_to_remove) > 0:
        #     docs_to_remove = list(set(docs_to_remove))
        #     doc_ids_list.append(doc_list[0])
        #     docs_to_remove_list.append(docs_to_remove)
        #     len_of_docs2remove_list.append(len(docs_to_remove))
        #     # results[(key1, key2)] = total_sum

        # index = HammingIndex()
        # # Define levels
        # index.set_distance_range("level_1", 0, 20)  # Example range for level_1
        # index.set_distance_range("level_2", 21, 40)  # Example range for level_2
        # index.set_distance_range("level_2", 21, 60)  # Example range for level_2
        #
        # index.add_documents(doc_minhashes)
        start_time = time.perf_counter_ns()  # Record start time in nanoseconds

        # std::cout << ">>>>>>>>>>>>START>>>>>>>>>>>>>>>>>>:" << indices.size()<< std::endl;

        # print(f">>>>>>>START>>>>>>{len(doc_list)}")

        while len(doc_list) > 1:
            docs_to_remove = []
            docs_to_remove_distance = []
            new_doc_list = []
            # this is the document we are going to keep
            first_doc = doc_list[0]
            first_mh = doc_minhashes[first_doc]
            print(f"=======<<<<<<<<<<<<<first_docfirst_docfirst_docfirst_doc<<<>>>>>>>>>>>>>>>>====:=======<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>====:first_doc {first_doc}")
            # print(np.array(first_mh).tolist())

            for int_id_column in doc_list[1:]:
                # print(f">>>> COMPARE WITH int_id_column {int_id_column}")

                # print(f">>>>>>>sec_doc>>>>>>{int_id_column}")
                doc_mh = doc_minhashes[int_id_column]
                distance_index = Murmur_MH.jaccard(np.array(first_mh), np.array(doc_mh))

                # # print(np.array(doc_mh).tolist())
                # #
                # # # Compute Jaccard similarity using XOR-based selection
                # # selected_set3 = self.xor_based_minhash_selection(set(first_mh), seed=0x12345678, k=20)
                # # selected_set4 = self.xor_based_minhash_selection(set(doc_mh), seed=0x12345678, k=20)
                # #
                # # # # Test the new approach on the identical MinHash sets
                # selected_set3 = self.xor_order_preserving_selection(set(first_mh), seed=0x12345678, k=20)
                # selected_set4 = self.xor_order_preserving_selection(set(doc_mh), seed=0x12345678, k=20)
                #
                # jaccard_xor1 = Murmur_MH.jaccard(np.array(selected_set3).tolist(), np.array(selected_set4).tolist())
                #
                # distance_index1 = self.jaccard_similarity(selected_set3, selected_set4)
                #
                # selected_set5 = self.hybrid_xor_hamming_selection(set(first_mh), seed=0x12345678, k=20)
                # selected_set6 = self.hybrid_xor_hamming_selection(set(doc_mh), seed=0x12345678, k=20)
                # jaccard_xor2 = Murmur_MH.jaccard(np.array(selected_set4).tolist(), np.array(selected_set6).tolist())
                #
                # distance_index2 = self.jaccard_similarity(selected_set5, selected_set6)
                #
                # print(f"=======<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>==jaccard_xor==111==: {jaccard_xor1} >>>>>>>>>>>>{distance_index}>>>>>>>>>>>> {self.jaccard_similarity(selected_set3,selected_set4)}")
                #
                # # Display results
                # print(f"=======<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>==jaccard_xor=111=: {jaccard_xor1} >>>>>>>>>>>>{distance_index}>>>>>>>>>>>> {distance_index1}")
                #
                # print(
                #     f"=======<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>==jaccard_xor=222=: {jaccard_xor2} >>>>>>>>>>>>>{distance_index}>>>>>>>>>>> {self.jaccard_similarity(selected_set5, selected_set6)}")
                #
                # # Display results
                # print(
                #     f"=======<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>==jaccard_xor=222=: {jaccard_xor2} >>>>>>>>>>>>>{distance_index}>>>>>>>>>>> {distance_index2}")
                # print(selected_set3)
                # print(selected_set3)
                # print(selected_set4)

                # distance_index = inverteFileIndex2.get_count(first_doc, int_id_column)
                # if distance_index == 0:
                # distance_index=self.index_query_comparison(first_mh, first_doc, doc_list[1:],inverteFileIndex2,data,
                #                           total_count=len(first_mh))
                # else:
                # print("=========INDEX VALUE1==========")
                # print(distance_index)
                if distance_index >= threshold:
                    # new_values = [first_doc,int_id_column,distance_index]
                    # self.append_to_jsonl( f'output_logs.jsonl', new_values)
                    docs_to_remove.append(int_id_column)
                    docs_to_remove_distance.append(distance_index)
                else:
                    new_doc_list.append(int_id_column)
            if len(docs_to_remove) > 0:

                docs_to_remove = list(set(docs_to_remove))
                docs_to_remove_distance = list(set(docs_to_remove_distance))

                doc_ids_list.append(first_doc)
                docs_to_remove_list.append(docs_to_remove)
                docs_to_remove_distance_list.append(docs_to_remove_distance)
                len_of_docs2remove_list.append(len(docs_to_remove))
            doc_list = new_doc_list

        # print(f">>>>>>>END>>>>>>{len(docs_to_remove)}")

        return doc_ids_list, docs_to_remove_list, len_of_docs2remove_list,docs_to_remove_distance_list

    def _jaccard_distance_calculation_1(self, row: List[pl.Series]) -> list[list]:
        # Process row and return a new list of Series or a new row
        threshold = self.jaccard_similarity_threshold
        doc_ids_list = []
        docs_to_remove_list = []
        len_of_docs2remove_list = []
        # sort documents
        document_data = row["document_data"]

        # # Sort the list by 'document_length'
        # sorted_document_data = sorted(document_data, key=lambda x: (-x["document_length"]))
        # # sorted_document_data = sorted(document_data, key=lambda x: (-x["document_length"], x["int_id_column"]))


        # Sort the list by 'document_length'
        # sorted_document_data = sorted(document_data, key=lambda x: (-x["document_length"], x["int_id_column"]))

        # Extracting int_id_column values into a list
        doc_list = [item["int_id_column"] for item in document_data]

        # Creating a dictionary with int_id_column as key and minhashes as value
        doc_minhashes = {item["int_id_column"]: item["minhashes"] for item in document_data}

        

        while len(doc_list) > 1:
            docs_to_remove = []
            new_doc_list = []
            # this is the document we are going to keep
            first_doc = doc_list[0]
            first_mh = doc_minhashes[first_doc]
            for int_id_column in doc_list[1:]:
                doc_mh = doc_minhashes[int_id_column]
                distance = Murmur_MH.jaccard(np.array(first_mh), np.array(doc_mh))
                if distance >= threshold:
                    docs_to_remove.append(int_id_column)
                else:
                    new_doc_list.append(int_id_column)
            if len(docs_to_remove) > 0:
                docs_to_remove = list(set(docs_to_remove))
                doc_ids_list.append(first_doc)
                docs_to_remove_list.append(docs_to_remove)
                len_of_docs2remove_list.append(len(docs_to_remove))
            doc_list = new_doc_list

        return doc_ids_list, docs_to_remove_list, len_of_docs2remove_list



    def get_clusters(self, band_segment_dataframe: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
        groupby_dataframe = band_segment_dataframe.group_by("band_hash").agg("document_data")
        cluster_dataframe = groupby_dataframe.with_columns(cluster_length=pl.col("document_data").list.len()).filter(
            pl.col("cluster_length") > 1
        )
        # self.logger.info(f"file_name = {file_name}")
        num_clusters = len(cluster_dataframe)
        if num_clusters > 0:
            sum_cdocs = cluster_dataframe.select(pl.sum("cluster_length")).item()
            max_cdocs = cluster_dataframe.select(pl.max("cluster_length")).item()
            min_cdocs = cluster_dataframe.select(pl.min("cluster_length")).item()
            avg_cdocs = cluster_dataframe.select(pl.mean("cluster_length")).item()
        else:
            sum_cdocs = 0
            max_cdocs = 0
            min_cdocs = 0
            avg_cdocs = 0
        self.logger.info(f"After GroupBy: {num_clusters} clusters with {sum_cdocs} total docs")
        self.logger.info(f" max/min/avg docs per cluster: {max_cdocs}/{min_cdocs}/{avg_cdocs:.2f}")
        cluster_stats = {
            "groupby_clusters": num_clusters,
            "cluster_duplicate_docs": sum_cdocs,
        }
        return cluster_dataframe, cluster_stats

    def analyze_clusters(self, df: pl.DataFrame,inverteFileIndex2,data,band) -> tuple[pl.DataFrame, dict[str, Any]]:
        # Define the schema with specific data types
        schema = {"first_doc": pl.Int64, "docs_to_remove": pl.List(pl.Int64),"docs_to_remove_distance": pl.List(pl.Float64), "docs_to_remove_length": pl.Int64}
        doc_ids_lists = []
        docs_to_remove_lists = []
        docs_to_remove_distance_lists = []

        len_of_docs2remove_lists = []
        i=0
        for row in df.iter_rows(named=True):
            # print(f">>>>>>>>Row {i}")
            doc_ids_list, docs_to_remove_list, len_of_docs2remove_list,docs_to_remove_distance_list = self.jaccard_distance_calculation(row,inverteFileIndex2,data,band)
            doc_ids_lists += doc_ids_list
            docs_to_remove_lists += docs_to_remove_list
            docs_to_remove_distance_lists += docs_to_remove_distance_list

            len_of_docs2remove_lists += len_of_docs2remove_list
        jaccard_cluster_dataframe = pl.DataFrame(
            {
                "first_doc": doc_ids_lists,
                "docs_to_remove": docs_to_remove_lists,
                "docs_to_remove_distance": docs_to_remove_distance_lists,
                "docs_to_remove_length": len_of_docs2remove_lists,
            },
            schema=schema,
        )
        filtered_jaccard_dataframe = jaccard_cluster_dataframe.filter(pl.col("docs_to_remove_length") > 0)
        num_clusters = len(filtered_jaccard_dataframe)
        if num_clusters > 0:
            sum_cdocs = filtered_jaccard_dataframe.select(pl.sum("docs_to_remove_length")).item()
            max_cdocs = filtered_jaccard_dataframe.select(pl.max("docs_to_remove_length")).item()
            min_cdocs = filtered_jaccard_dataframe.select(pl.min("docs_to_remove_length")).item()
            avg_cdocs = filtered_jaccard_dataframe.select(pl.mean("docs_to_remove_length")).item()
        else:
            sum_cdocs = 0
            max_cdocs = 0
            min_cdocs = 0
            avg_cdocs = 0
        self.logger.info(f"After Jaccard: {num_clusters} clusters with {sum_cdocs} total docs")
        self.logger.info(f" max/min/avg docs per cluster: {max_cdocs}/{min_cdocs}/{avg_cdocs:.2f}")
        jaccard_stats = {
            "jaccard_clusters": num_clusters,
            "jaccard_duplicate_docs": sum_cdocs,
        }
        if self.sort_output:
            filtered_jaccard_dataframe = filtered_jaccard_dataframe.sort(by="first_doc")
        return filtered_jaccard_dataframe, jaccard_stats

    def sanitize_folder_name(self, folder_name: str) -> str:
        if "://" in folder_name:
            _, folder_name = folder_name.split("://")
        if folder_name[-1] != "/":
            folder_name = f"{folder_name}/"
        return folder_name

    def consolidate_band_segment_files(self, files: dict[str, bytes]) -> tuple[pl.DataFrame, dict[str, Any]]:
        band_segment_dataframe = pl.DataFrame()
        total_input_rows = 0
        for fname, contents in files.items():
            df = pl.read_parquet(io.BytesIO(contents))
            total_input_rows += len(df)
            self.logger.debug(f"{fname} has {len(df)} rows")
            band_segment_dataframe = band_segment_dataframe.vstack(df)

        consolidation_stats = {
            "input_files": len(files),
            "input_bytes": sum(len(v) for v in files.values()),
            "input_rows": total_input_rows,
            "consolidated_files": 1,
            "consolidated_bytes": band_segment_dataframe.to_arrow().nbytes,
            "consolidated_rows": len(band_segment_dataframe),
        }
        return band_segment_dataframe, consolidation_stats

class ClusterAnalysisTransformConfiguration(TransformConfiguration):

    """
    Provides support for configuring and using the associated Transform class include
    configuration with CLI args.
    """

    def __init__(self):
        super().__init__(
            name=short_name,
            transform_class=ClusterAnalysisTransform,
            remove_from_metadata=[],
        )
        self.logger = get_logger(__name__, level="INFO")

    def add_input_params(self, parser: ArgumentParser) -> None:
        """
        Add Transform-specific arguments to the given  parser.
        This will be included in a dictionary used to initialize the NOOPTransform.
        By convention a common prefix should be used for all transform-specific CLI args
        (e.g, noop_, pii_, etc.)
        """
        parser.add_argument(
            f"--{jaccard_similarity_threshold_cli_param}",
            type=float,
            default=jaccard_similarity_threshold_default,
            help="Jaccard similarity threshold above which two documents are duplicates",
        )
        parser.add_argument(
            f"--{num_bands_cli_param}",
            type=int,
            default=num_bands_default,
            help="The number of bands used in the banding technique",
        )
        parser.add_argument(
            f"--{num_segments_cli_param}",
            type=int,
            default=num_segments_default,
            help="The number of segments dividing the hashing space for each band",
        )
        parser.add_argument(
            f"--{sort_output_cli_param}",
            type=bool,
            default=sort_output_default,
            help="Sort the similarity clusters by the document ID of the kept doc (used primarily for testing)",
        )

    def apply_input_params(self, args: Namespace) -> bool:
        """
        Validate and apply the arguments that have been parsed
        :param args: user defined arguments.
        :return: True, if validate pass or False otherwise
        """
        captured = CLIArgumentProvider.capture_parameters(args, cli_prefix, False)
        self.params = self.params | captured
        self.logger.info(f"{short_name} parameters are : {self.params}")
        return True
