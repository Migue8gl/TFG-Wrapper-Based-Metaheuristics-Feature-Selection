import csv
from typing import Optional

import arff
import numpy as np
import pandas as pd
from .constants import LABELS, SAMPLE


def load_data(file_path: str) -> Optional[np.ndarray]:
    """
    Loads data from a file, supporting both CSV and ARFF formats.

    Args:
       file_path (str): The path to the file.

    Returns:
       data (numpy.ndarray, None): The loaded data as a numpy array, or None if an error occurred.
    """
    try:
        # Check the file extension
        if file_path.endswith('.arff'):
            with open(file_path, 'r') as arff_file:
                dataset = arff.load(arff_file, encode_nominal=True)
                df = pd.DataFrame(dataset['data'])

                # Convert unknown values to NaN
                df.replace('?', np.nan, inplace=True)

                for col in df.columns:
                    df.fillna({col: df[col].mean()}, inplace=True)

                # Transform all columns except the last one to float64
                data = df.to_numpy()
                data[:, :-1] = data[:, :-1].astype(np.float64)

        elif file_path.endswith('.csv'):
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                data_list = [row for row in csv_reader]
                data = np.array(data_list)
        else:
            print(
                "Unsupported file format. Please provide a .arff or .csv file."
            )
            return None

        return data

    except Exception as e:
        print(f"An error occurred while loading the data: {str(e)}")
        return None


def split_data_to_dict(dataset: np.ndarray) -> dict:
    """
    Splits the given dataset into a dictionary containing the samples and labels.

    Args:
       dataset (numpy.ndarray): The dataset to be split.

    Returns:
       data (dict): A dictionary containing the samples and labels.
    """
    np.random.shuffle(dataset)
    samples = dataset[:, :-1].astype(np.float64)
    classes = dataset[:, -1].astype(np.int8)

    return {SAMPLE: samples, LABELS: classes}
