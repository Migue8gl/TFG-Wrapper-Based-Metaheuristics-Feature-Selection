import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import constants

del sys.path[0]

# Initialize an empty list to store individual DataFrames
dfs = []

# Iterate over directories in RESULTS_DIR
for dataset_dir in os.listdir(constants.RESULTS_DIR):
    dataset_path = os.path.join(constants.RESULTS_DIR, dataset_dir)
    if os.path.isdir(dataset_path):
        # Iterate over CSV files in each directory
        for filename in os.listdir(dataset_path):
            if filename.endswith(
                    ".csv") and filename != 'analysis_results.csv':
                # Extract dataset name and optimizer name from the filename
                dataset_name, optimizer_name = filename.split('_')[:2]

                df = pd.read_csv(os.path.join(dataset_path, filename))

                # Add columns for dataset name and optimizer name
                df['dataset'] = dataset_name
                df['optimizer'] = optimizer_name[:
                                                 -4]  # Remove ".csv" extension from optimizer name
                # Round columns to a maximum of 3 decimal places
                df = df.round({
                    'best': 3,
                    'avg': 3,
                    'std_dev': 3,
                    'acc': 3,
                    'n_features': 3,
                    'selected_rate': 3,
                    'execution_time': 3
                })

                dfs.append(df)

# Reorder columns
combined_data = pd.concat(dfs)[[
    'classifier', 'dataset', 'optimizer', 'best', 'avg', 'std_dev', 'acc',
    'n_features', 'selected_rate', 'execution_time'
]].to_csv(os.path.join(constants.RESULTS_DIR, 'analysis_results.csv'),
          index=False)
