import os
import pandas as pd

# Directory containing the CSV files
directory = 'results/'

# Initialize an empty list to store individual DataFrames
dfs = []

for filename in os.listdir(directory):
    if filename.endswith(".csv") and filename != 'analysis_results.csv':
        # Extract dataset name and optimizer name from the filename
        dataset_name, optimizer_name = filename.split('_')[:2]

        df = pd.read_csv(os.path.join(directory, filename))

        # Add columns for dataset name and optimizer name
        df['Dataset'] = dataset_name
        df['Optimizer'] = optimizer_name[:
                                         -4]  # Remove ".csv" extension from optimizer name

        dfs.append(df)
        combined_data = pd.concat(dfs, ignore_index=True)

# Reorder columns
combined_data = combined_data[[
    'Dataset', 'Optimizer', 'Best', 'Avg', 'StdDev'
]]

# Write the combined data to a new CSV file
combined_data.to_csv(directory + 'analysis_results.csv', index=False)
