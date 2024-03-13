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
        df['dataset'] = dataset_name
        df['optimizer'] = optimizer_name[:
                                         -4]  # Remove ".csv" extension from optimizer name

        dfs.append(df)
        combined_data = pd.concat(dfs)

# Reorder columns
combined_data = combined_data[[
    'classifier', 'dataset', 'optimizer', 'best', 'avg', 'std_dev'
]]

# Write the combined data to a new CSV file
combined_data.to_csv(directory + 'analysis_results.csv', index=False)
