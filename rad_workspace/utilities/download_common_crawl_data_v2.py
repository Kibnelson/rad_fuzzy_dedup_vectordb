import os
import requests
import gzip
import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing


def download_wet_paths(dataset_name):
    """
    Downloads the `wet.paths.gz` file for a specified dataset and extracts it.

    Args:
        dataset_name (str): The name of the Common Crawl dataset (e.g., "CC-MAIN-2024-30").

    Returns:
        str: Path to the extracted `wet.paths` file.
    """
    base_url = f"https://data.commoncrawl.org/crawl-data/{dataset_name}/wet.paths.gz"
    output_dir = dataset_name
    os.makedirs(output_dir, exist_ok=True)
    gz_file_path = os.path.join(output_dir, "wet.paths.gz")
    extracted_file_path = os.path.join(output_dir, "wet.paths")

    print(f"Downloading {base_url}...")
    response = requests.get(base_url, stream=True)
    response.raise_for_status()

    with open(gz_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Extracting {gz_file_path}...")
    with gzip.open(gz_file_path, 'rb') as gz_file:
        with open(extracted_file_path, 'wb') as extracted_file:
            shutil.copyfileobj(gz_file, extracted_file)

    os.remove(gz_file_path)
    print(f"Extracted to {extracted_file_path}")
    return extracted_file_path


def download_and_unzip_file(file_url, output_dir):
    """
    Downloads and unzips a single WET file.

    Args:
        file_url (str): Full URL of the WET file.
        output_dir (str): Directory to save the downloaded and unzipped file.

    Returns:
        str: Path to the unzipped WET file, or None if an error occurred.
    """
    try:
        file_name = os.path.basename(file_url)
        gz_file_path = os.path.join(output_dir, file_name)
        unzipped_file_path = gz_file_path.replace(".gz", "")

        # Download the .gz file
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        with open(gz_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Unzip the file
        with gzip.open(gz_file_path, 'rb') as gz_file:
            with open(unzipped_file_path, 'wb') as out_file:
                shutil.copyfileobj(gz_file, out_file)

        os.remove(gz_file_path)
        return unzipped_file_path
    except Exception as e:
        print(f"Failed to process {file_url}: {e}")
        return None


def parse_warc_with_language_filter(file_path, required_language):
    records = []
    with open(file_path, 'r') as f:
        record = {}
        content = []
        inside_record = False
        for line in f:
            line = line.strip()
            if line.startswith("WARC-Target-URI:"):
                record["url"] = line.split(":", 1)[1].strip()
            elif line.startswith("WARC-Date:"):
                record["date"] = line.split(":", 1)[1].strip()
            elif line.startswith("WARC-Record-ID:"):
                record["record_id"] = line.split(":", 1)[1].strip().replace("<urn:uuid:", "").replace(">", "")
            elif line.startswith("WARC-Refers-To:"):
                record["refers_to"] = line.split(":", 1)[1].strip()
            elif line.startswith("WARC-Block-Digest:"):
                record["block_digest"] = line.split(":", 1)[1].strip()
            elif line.startswith("WARC-Identified-Content-Language:"):
                record["language"] = line.split(":", 1)[1].strip()
            elif line.startswith("Content-Type:"):
                record["content_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("Content-Length:"):
                inside_record = True
                content = []
            elif line.startswith("WARC/1.0"):
                if record and content:
                    record["content"] = "\n".join(content).strip()
                    if record.get("language") == required_language:
                        records.append(record)
                record = {}
                content = []
                inside_record = False
            elif inside_record:
                content.append(line)

        if record and content:
            record["content"] = "\n".join(content).strip()
            if record.get("language") == required_language:
                records.append(record)

    return records


def save_to_parquet(records, output_file):
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(output_file, index=False)
        print(f"Saved {output_file}")
    else:
        print(f"No records to save for {output_file}")


def process_file(file_path, output_dir, required_language):
    try:
        print(f"Processing {file_path}...")
        records = parse_warc_with_language_filter(file_path, required_language)
        output_file = os.path.join(output_dir, os.path.basename(file_path).replace(".warc.wet", ".parquet"))
        save_to_parquet(records, output_file)
        os.remove(file_path)  # Remove the WET file after processing
        return f"Successfully processed {file_path}"
    except Exception as e:
        return f"Failed to process {file_path}: {e}"


def process_wet_files(dataset_name, required_language, start=0, end=None, max_workers=None):
    # Create dataset-specific directories
    base_dir = dataset_name
    os.makedirs(base_dir, exist_ok=True)
    wet_files_dir = os.path.join(base_dir, "wet_files")
    parquet_dir = os.path.join(base_dir, "parquet_files")
    os.makedirs(wet_files_dir, exist_ok=True)
    os.makedirs(parquet_dir, exist_ok=True)

    # Step 1: Download and extract `wet.paths`
    paths_file = download_wet_paths(dataset_name)

    # Step 2: Read the specified range of file paths
    with open(paths_file, 'r') as f:
        file_paths = [line.strip() for line in f if line.strip()]
    if end is None or end > len(file_paths):
        end = len(file_paths)
    file_paths = file_paths[start:end]

    base_url = "https://data.commoncrawl.org"

    if max_workers is None:
        max_workers = max(1, int(multiprocessing.cpu_count() * 0.8))

    # Step 3: Download and unzip the files in the range
    print(f"Downloading and unzipping WET files from index {start} to {end - 1}...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_and_unzip_file, f"{base_url}/{path}", wet_files_dir): path for path in file_paths}
        unzipped_files = [future.result() for future in as_completed(futures) if future.result()]

    # Step 4: Process WET files and save as Parquet
    print("Processing WET files into Parquet format...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file_path, parquet_dir, required_language): file_path for file_path in unzipped_files}
        for future in as_completed(futures):
            print(future.result())


if __name__ == '__main__':
    dataset_name = input("Enter the dataset name (e.g., CC-MAIN-2024-30): ").strip()
    required_language = input("Enter the required language (e.g., eng, deu): ").strip()
    start = int(input("Enter the start index (e.g., 0): ").strip())
    end = int(input("Enter the end index (e.g., 1000): ").strip())
    process_wet_files(dataset_name, required_language, start=start, end=end)

