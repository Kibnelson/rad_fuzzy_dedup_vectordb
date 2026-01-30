import pandas as pd

# Define the input and output file paths
input_file = 'train-00001-of-00005.parquet'  # Replace with your Parquet file path
output_file = '12.parquet'  # Desired output file path

#lm1b_all.parquet
input_file ="c410M.parquet"
output_file = 'c4_3m_withid.parquet'  # Desired output file path

# Read the existing Parquet file into a DataFrame
df = pd.read_parquet(input_file)
# df.rename(columns={'int_id_column2_old': 'int_id_column2_old2'}, inplace=True)

# Add a new column with incremental numbers starting from 1
# df['int_id_column'] = range(1, len(df) + 1)
df['int_id_column'] = range(len(df))
# Rename the column (replace 'old_column_name' with the actual name)
df.rename(columns={'text': 'contents'}, inplace=True)

# Write the modified DataFrame back to a new Parquet file
df.to_parquet(output_file, index=False)

print(f"Added incremental column and saved to {output_file} successfully.")
