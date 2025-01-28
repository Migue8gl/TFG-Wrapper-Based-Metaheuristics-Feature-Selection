import os

import constants
import pandas as pd


def process_csv_file(file_path: str):
    """
    Adds a new column 'dataset' and 'optimizer' based on the file name to the csv readed.

    Args:
        file_path (str): Name of the file path where csv is.

    Returns:
        df (pd.DataFrame): DataFrame processed.
    """
    dataset_name, optimizer_name = os.path.basename(file_path).split("_")[:2]
    df = pd.read_csv(file_path)
    df["dataset"] = dataset_name
    df["optimizer"] = optimizer_name[:-4]  # Remove ".csv" extension from optimizer name
    return df


def generate_analysis_results(encoding: str):
    """
    Generates a csv with the analysis results of all csv files in a directory.

    Args:
        encoding (str): Encoding of the data. It can be 'binary' or 'real'.
    """
    dfs = []
    encoding_dir = os.path.join(constants.RESULTS_DIR, encoding)
    for dataset_dir in os.listdir(encoding_dir):
        dataset_path = os.path.join(encoding_dir, dataset_dir)
        if os.path.isdir(dataset_path):
            for filename in os.listdir(dataset_path):
                if (
                    filename.endswith(".csv")
                    and filename != "analysis_results.csv"
                    and "all_fitness" not in filename
                ):
                    file_path = os.path.join(dataset_path, filename)
                    dfs.append(process_csv_file(file_path))

    combined_data = pd.concat(dfs)[
        [
            "classifier",
            "dataset",
            "optimizer",
            "all_fitness",
            "best",
            "avg",
            "std_dev",
            "acc",
            "acc_std_dev"
            "n_features",
            "n_features_std_dev",
            "selected_rate",
            "selected_rate_std_dev",
            "execution_time",
        ]
    ]

    combined_data.to_csv(
        os.path.join(encoding_dir, "analysis_results.csv"), index=False
    )


def main():
    for encoding in ["binary", "real"]:
        generate_analysis_results(encoding)


if __name__ == "__main__":
    main()
