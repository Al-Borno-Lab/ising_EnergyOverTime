import sys
import pandas as pd

# Usage: python transpose_csv.py input.csv output.csv

if len(sys.argv) != 3:
    print("Usage: python transpose_csv.py input.csv output.csv")
    sys.exit(1)

input_csv = sys.argv[1]
output_csv = sys.argv[2]

# Read the CSV
df = pd.read_csv(input_csv)

# Identify the folder_level columns (all columns starting with 'folder_level_')
id_vars = [col for col in df.columns if col.startswith('folder_level_')]

# All other columns are assumed to be time columns
value_vars = [col for col in df.columns if col not in id_vars]

# Melt the dataframe
df_melted = df.melt(id_vars=id_vars, value_vars=value_vars,
                    var_name='Time', value_name='energy_at_time')

# Save to output
df_melted.to_csv(output_csv, index=False)
print(f"Transposed CSV written to {output_csv}") 