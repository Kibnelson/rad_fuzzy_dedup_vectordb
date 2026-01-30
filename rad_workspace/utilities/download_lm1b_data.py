import tensorflow_datasets as tfds
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow_datasets as tfds
print("tfds file:", tfds.__file__)
print("tfds version:", tfds.__version__)


# Specify the cache/download directory for TensorFlow Datasets.
data_dir = "./data/lm1b/tensorflow_datasets"

# Load LM1B training split, using the specified data directory.
ds = tfds.load('lm1b', split='train', as_supervised=False, data_dir=data_dir)

output_file = "lm1b_all.parquet"

# Set the limit to 3 million examples
limit = 30000000
chunk_size = 1000  # Adjust based on your available memory

batch = []
writer = None

for count, example in enumerate(ds.as_numpy_iterator(), 1):
    # Stop when we've reached our limit
    #if count > limit:
    #    break

    # Decode the 'text' field from bytes to a UTF-8 string.
    if isinstance(example["text"], bytes):
        example["text"] = example["text"].decode('utf-8')
    
    batch.append(example)
    
    if len(batch) == chunk_size:
        # Convert the current batch (a list of dicts) into a PyArrow Table
        table = pa.Table.from_pydict({
            key: [ex[key] for ex in batch]
            for key in batch[0]
        })
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
        batch = []  # Reset the batch for the next chunk
        
        if count % 100000 == 0:
            print(f"Processed {count} examples...")

# Write any remaining examples that didn't fill a complete chunk
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

print(f"Downloaded {limit} examples from LM1B and saved to {output_file}")

