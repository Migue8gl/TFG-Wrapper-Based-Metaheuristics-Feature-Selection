import arff
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from constants import *
from sklearn.preprocessing import StandardScaler
from typing import Optional

# ---------------------------------- DATA ------------------------------------ #


def load_arff_data(file_path: str) -> Optional[np.ndarray]:
    """
    Loads ARFF data from a file.

    Parameters:
        - file_path (str): The path to the ARFF file.

    Returns:
        - data (numpy.ndarray): The loaded ARFF data as a numpy array, or None if an error occurred.
    """
    try:
        with open(file_path, 'r') as arff_file:
            dataset = arff.load(arff_file)
            data = np.array(dataset['data'])

            # Transform all columns except the last one to float64
            data[:, :-1] = data[:, :-1].astype(np.float64)

        return data
    except Exception as e:
        print(f"An error occurred while loading the ARFF data: {str(e)}")
        return None


def split_data_to_dict(dataset: np.narray) -> dict:
    """
    Splits the given dataset into a dictionary containing the samples and labels.

    Parameters:
    - dataset (numpy.ndarray): The dataset to be split.

    Returns:
    - data (dict): A dictionary containing the samples and labels.
    """
    np.random.shuffle(dataset)
    samples = dataset[:, :-1].astype(np.float64)
    classes = dataset[:, -1]

    return {SAMPLE: samples, LABELS: classes}


def split_data_to_dict_train_test(dataset: np.narray,
                                  train_ratio: float = 0.8) -> dict:
    """
    Splits the given dataset into a dictionary containing the samples and labels for training and testing.

    Parameters:
    - dataset (numpy.ndarray): The dataset to be split.
    - train_ratio (float): The ratio of data to be used for training. Default is 0.8.

    Returns:
    - data (dict): A dictionary containing the training and testing samples and labels.
    """
    np.random.shuffle(dataset)

    # Split the dataset into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        dataset[:, :-1].astype(np.float64),  # Features
        dataset[:, -1],  # Labels
        train_size=train_ratio,  # Ratio for training data
        random_state=42  # Random seed for reproducibility
    )

    return {
        "train": {
            SAMPLE: train_data,
            LABELS: train_labels
        },
        "test": {
            SAMPLE: test_data,
            LABELS: test_labels
        }
    }


def split_data(dataset: np.narray) -> tuple:
    """
    Split the given dataset into samples and classes.

    Parameters:
    - dataset (numpy.ndarray): The dataset to be split.

    Returns:
    - data (tuple): A tuple containing two numpy.ndarrays. The first array contains the samples, and the second array contains the classes.
    """
    samples = dataset[:, :-1].astype(np.float64)
    classes = dataset[:, -1]

    return samples, classes


def split_dicts_keys_to_lists(list_of_dicts: list) -> tuple:
    """
    Extracts values from two keys in a list of dicts.

    Parameters:
    - list_of_dicts (list): List of dictionaries with two keys.

    Returns:
    -  values (tuple): Two lists, one with values from the first key and another with values from the second key.
    """
    keys = list(list_of_dicts[0].keys())
    first_key_values = [item[keys[0]] for item in list_of_dicts]
    second_key_values = [item[keys[1]] for item in list_of_dicts]

    return first_key_values, second_key_values


# ---------------------------- NORMALIZE / SCALING DATA ------------------------------ #
# https://en.wikipedia.org/wiki/Normalization_(statistics)


def scaling_min_max(data: np.narray) -> np.narray:
    """
    Normalize the features of a dataset using the MinMaxScaler.

    Parameters:
    - data (numpy.ndarray): The input dataset.

    Returns:
    - normalized_data (numpy.ndarray): The normalized dataset.
    """
    # Separate the features (x) and labels (y)
    x, y = split_data(data)

    scaler = MinMaxScaler()

    # Fit the scaler to the features and normalize the data between 0 and 1
    x_normalized = scaler.fit_transform(x)

    # Combine the normalized features (x_normalized) and labels (y) into a single array
    return np.column_stack((x_normalized, y))


def scaling_std_score(data: np.narray) -> np.narray:
    """
    Scale the input data using the standard score method.

    Parameters:
    - data (numpy.ndarray): Input data array containing both features and labels.

    Returns:
    - scaled_data (numpy.ndarray): The scaled data array.
    """
    # Separate the features (x) and labels (y)
    # Separate the features (x) and labels (y)
    x = data[:, :-1]
    y = data[:, -1]

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the features
    x_scaled = scaler.fit_transform(x)

    # Combine the normalized features (x_scaled) and labels (y) into a single array
    return np.column_stack((x_scaled, y))
