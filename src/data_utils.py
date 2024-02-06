import arff
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from constants import *

# ---------------------------------- DATA GENERAL ------------------------------------ #


def load_arff_data(file_path):
    """
    Loads ARFF data from a file.

    Parameters:
    - file_path (str): The path to the ARFF file.

    Returns:
    - numpy.ndarray or None: The loaded ARFF data as a numpy array, or None if an error occurred.
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


def split_data_to_dict(dataset):
    """
    Splits the given dataset into a dictionary containing the samples and labels.

    Parameters:
    - dataset (numpy.ndarray): The dataset to be split.

    Returns:
    - dict: A dictionary containing the samples and labels.
    """
    samples = dataset[:, :-1].astype(np.float64)
    classes = dataset[:, -1]
    return {DATA: samples, LABELS: classes}


def split_data(dataset):
    """
    Split the given dataset into samples and classes.

    Parameters:
    - dataset (numpy.ndarray): The dataset to be split.

    Returns:
    - tuple: A tuple containing two numpy.ndarrays. The first array contains the samples, and the second array contains the classes.
    """
    samples = dataset[:, :-1].astype(np.float64)
    classes = dataset[:, -1]
    return samples, classes


def split_dicts_keys(list_of_dicts):
    """
    Extracts values from two keys in a list of dicts.

    Parameters:
    - list_of_dicts (list): List of dictionaries with two keys.

    Returns:
    - Two lists, one with values from the first key and another with values from the second key.
    """
    keys = list(list_of_dicts[0].keys())
    first_key_values = [item[keys[0]] for item in list_of_dicts]
    second_key_values = [item[keys[1]] for item in list_of_dicts]

    return first_key_values, second_key_values


# ---------------------------- NORMALIZE / SCALING DATA ------------------------------ #
# https://en.wikipedia.org/wiki/Normalization_(statistics)


def scaling_min_max(data):
    """
    Normalize the features of a dataset using the MinMaxScaler.

    Parameters:
    - data (numpy.ndarray): The input dataset.

    Returns:
    - numpy.ndarray: The normalized dataset with features and labels combined.
    """
    # Separate the features (x) and labels (y)
    x, y = split_data(data)

    scaler = MinMaxScaler()

    # Fit the scaler to the features and normalize the data between 0 and 1
    x_normalized = scaler.fit_transform(x)

    # Combine the normalized features (x_normalized) and labels (y) into a single array
    return np.column_stack((x_normalized, y))


def scaling_standard_score(data):
    """
    Scale the input data using the standard score method.

    Parameters:
    - data (ndarray): Input data array containing both features and labels.

    Returns:
    - ndarray: The scaled data array with normalized features and labels.
    """
    # Separate the features (x) and labels (y)
    x, y = split_data(data)

    # Scale data to meet std deviation of 0 and mean 1
    x_scaled = (np.mean(x, axis=0) - x) / np.std(x, axis=0)

    # Combine the normalized features (x_normalized) and labels (y) into a single array
    return np.column_stack((x_scaled, y))


# ------------------------------ OPTIMIZERS -------------------------------- #


def get_optimizer_parameters(optimizer=None, solution_len=2):
    optimizer_title = 'No optimizer Selected'
    parameters = {}

    optimizer_upper = optimizer.upper()

    if optimizer_upper == 'GOA':
        parameters = {
            'grasshoppers': DEFAULT_POPULATION_SIZE,
            'iterations': DEFAULT_ITERATIONS,
            'min_values': [0] * (solution_len),
            'max_values': [1] * (solution_len),
            'binary': 's',  # Best binary version in paper
        }
        optimizer_title = 'Running GOA'
    elif optimizer_upper == 'DA':
        parameters = {
            'size': DEFAULT_POPULATION_SIZE,
            'generations': DEFAULT_ITERATIONS,
            'min_values': [0] * (solution_len),
            'max_values': [1] * (solution_len),
            'binary': 's',  # Binary version proposed in paper
        }
        optimizer_title = 'Running DA'
    elif optimizer_upper == 'GWO':
        parameters = {
            'pack_size': DEFAULT_POPULATION_SIZE,
            'iterations': DEFAULT_ITERATIONS,
            'min_values': [0] * (solution_len),
            'max_values': [1] * (solution_len),
            'binary': 's',  # Best binary version in the paper
        }
        optimizer_title = 'Running GWO'
    elif optimizer_upper == 'WOA':
        parameters = {
            'hunting_party': DEFAULT_POPULATION_SIZE,
            'iterations': DEFAULT_ITERATIONS,
            'min_values': [0] * (solution_len),
            'max_values': [1] * (solution_len),
            'spiral_param': 1,
            'binary': 's',
        }
        optimizer_title = 'Running WOA'
    elif optimizer_upper == 'ABCO':
        parameters = {
            'food_sources': DEFAULT_POPULATION_SIZE,
            'iterations': DEFAULT_ITERATIONS,
            'min_values': [0] * (solution_len),
            'max_values': [1] * (solution_len),
            'employed_bees': 3,
            'outlookers_bees': 3,
            'limit': 3,
            'binary': 's',
        }
        optimizer_title = 'Running ABCO'
    elif optimizer_upper == 'BA':
        parameters = {
            'swarm_size': DEFAULT_POPULATION_SIZE,
            'iterations': DEFAULT_ITERATIONS,
            'min_values': [0] * (solution_len),
            'max_values': [1] * (solution_len),
            'alpha': 0.9,
            'gama': 0.9,
            'fmin': 0,
            'fmax': 10,
            'binary': 's',
        }
        optimizer_title = 'Running BA'
    elif optimizer_upper == 'PSO':
        parameters = {
            'swarm_size': DEFAULT_POPULATION_SIZE,
            'iterations': DEFAULT_ITERATIONS,
            'min_values': [0] * (solution_len),
            'max_values': [1] * (solution_len),
            'decay': 0,
            'w': 0.9,
            'c1': 2,
            'c2': 2,
            'verbose': True,
            'binary': 's',
        }
        optimizer_title = 'Running PSO'
    elif optimizer_upper == 'FA':
        parameters = {
            'swarm_size': DEFAULT_POPULATION_SIZE,
            'generations': DEFAULT_ITERATIONS,
            'min_values': [0] * (solution_len),
            'max_values': [1] * (solution_len),
            'alpha_0': 0.02,
            'beta_0': 0.1,
            'gama': 1,
            'verbose': True,
            'binary': 's',
        }
        optimizer_title = 'Running PSO'
    elif optimizer_upper == 'GA':
        parameters = {
            'population_size': DEFAULT_POPULATION_SIZE,
            'generations': DEFAULT_ITERATIONS,
            'min_values': [0] * (solution_len),
            'max_values': [1] * (solution_len),
            'crossover_rate': 1,
            'mutation_rate': 0.05,
            'elite': 3,
            'verbose': True,
        }
        optimizer_title = 'Running GA'

    return parameters, optimizer_title


def get_optimizers_list():
    return list(OPTIMIZERS.keys())


def get_optimizer(optimizer):
    return OPTIMIZERS[optimizer.upper()]


def get_optimizer_name_by_function(optimizer_func):
    for key, value in OPTIMIZERS.items():
        if value == optimizer_func:
            return key
