from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

# Specify the cache/download directory for Hugging Face Datasets.
cache_dir = "./data/tensorflow_datasets"

# Load the C4 dataset in streaming mode using the English configuration.
# Pass trust_remote_code=True to allow custom code execution.
dataset = load_dataset("c4", "en", split="train", streaming=True, cache_dir=cache_dir, trust_remote_code=True)

output_file = "c4_364K_validation.parquet"
limit = 30000000
chunk_size = 1000  # Adjust based on available memory

batch = []
writer = None

for count, example in enumerate(dataset, 1):
    if count > limit:
        break
    batch.append(example)
    
    if len(batch) == chunk_size:
        # Convert the current batch (list of dicts) to a PyArrow Table.
        table = pa.Table.from_pydict({
            key: [ex[key] for ex in batch]
            for key in batch[0]
        })
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
        batch = []  # Reset the batch
        
        if count % 100000 == 0:
            print(f"Processed {count} examples...")

# Write any remaining examples in the final incomplete batch.
if batch:
    table = pa.Table.from_pydict({
        key: [ex[key] for ex in batch]
        for key in batch[0]
    })
    if writer is None:
        writer = pq.ParquetWriter(output_file, table.schema)
    writer.write_table(table)

if writer:
    writer.close()

print(f"Saved {limit} examples from C4 to {output_file}")