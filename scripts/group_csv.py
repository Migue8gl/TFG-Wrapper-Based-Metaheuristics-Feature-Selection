import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import constants

del sys.path[0]


def process_csv_file(file_path):
    dataset_name, optimizer_name = os.path.basename(file_path).split('_')[:2]
    df = pd.read_csv(file_path)
    df['dataset'] = dataset_name
    df['optimizer'] = optimizer_name[:
                                     -4]  # Remove ".csv" extension from optimizer name
    return df.round({
        'best': 3,
        'avg': 3,
        'std_dev': 3,
        'acc': 3,
        'n_features': 3,
        'selected_rate': 3,
        'execution_time': 3
    })


def generate_analysis_results(encoding):
    dfs = []
    encoding_dir = os.path.join(constants.RESULTS_DIR, encoding)
    for dataset_dir in os.listdir(encoding_dir):
        dataset_path = os.path.join(encoding_dir, dataset_dir)
        if os.path.isdir(dataset_path):
            for filename in os.listdir(dataset_path):
                if filename.endswith(
                        ".csv") and filename != 'analysis_results.csv':
                    file_path = os.path.join(dataset_path, filename)
                    dfs.append(process_csv_file(file_path))

    combined_data = pd.concat(dfs)[[
        'classifier', 'dataset', 'optimizer', 'best', 'avg', 'std_dev', 'acc',
        'n_features', 'selected_rate', 'execution_time'
    ]]

    combined_data.to_csv(os.path.join(encoding_dir, 'analysis_results.csv'),
                         index=False)


def main():
    for encoding in ['binary', 'real']:
        generate_analysis_results(encoding)


if __name__ == "__main__":
    main()
