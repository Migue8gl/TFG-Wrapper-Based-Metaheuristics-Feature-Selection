import time
from typing import Optional

import numpy as np
from constants import DATA, DEFAULT_FOLDS, DEFAULT_ITERATIONS, LABELS, SAMPLE
from optimizer import Optimizer
from data_utils import split_data_to_dict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)


def calculate_average_metric(metrics_each_fold: dict, key: str) -> list:
    """
    Calculate the average metric values from a dictionary of metrics values for each fold.

    Parameters:
        - metrics_each_fold (dict): Dictionary of metrics values for each fold.
        - key (str): Key specifying the values to extract.

    Returns:
        - average_fitness_values (list): An array of average fitness values across all folds.
    """
    # Transpose fitness values to have each list represent values for a specific index
    transposed_fitness = np.array([[item[key] for item in sublist]
                                   for sublist in metrics_each_fold.values()
                                   ]).T

    # Calculate mean of each index across all lists
    average_metrics_values = np.mean(transposed_fitness, axis=1)

    return average_metrics_values


def k_fold_cross_validation(optimizer: object,
                            dataset: Optional[dict] = None,
                            k: int = DEFAULT_FOLDS,
                            scaler: int = 1,
                            verbose: bool = False) -> dict:
    """
    Implementation of k-fold cross-validation.

    Parameters:
        - optimizer (object): The optimizer to be used.
        - dataset (dict, optional): The dataset in dict form.
        - k (int, optional): The number of folds.
        - scaler (int, optional): The type of scaling to be used.
        - verbose (bool, optional): Whether to print the results.

    Returns:
        - metrics (dict): The dictionary containing metrics of the results.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    test_fitness = []
    test_accuracy = []
    test_selected_features = []
    test_selected_rate = []
    metrics_each_fold = {}
    fold_index = 0
    execution_time = 0

    for train_index, test_index in skf.split(dataset[SAMPLE], dataset[LABELS]):
        x_train, x_test = dataset[SAMPLE][train_index], dataset[SAMPLE][
            test_index]
        y_train, y_test = dataset[LABELS][train_index], dataset[LABELS][
            test_index]

        if scaler == 1:
            scaler = MinMaxScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            train = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)
            test = np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1)
        elif scaler == 2:
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            train = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)
            test = np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1)
        else:
            train = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)
            test = np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1)

        sample = split_data_to_dict(train)
        sample_test = split_data_to_dict(test)

        # Run optimization algorithm on the current fold
        start_time = time.time()
        result, metrics_values = optimizer.optimize(sample)
        execution_time += time.time() - start_time
        metrics_each_fold[fold_index] = metrics_values

        # Evaluate the model on the test set of the current fold
        optimizer.params['target_function_parameters'][DATA] = sample_test
        optimizer.params['target_function_parameters']['weights'] = result[:-4]

        metrics = Optimizer.fitness(
            **optimizer.params['target_function_parameters'])

        test_fitness.append(metrics['fitness'])
        test_accuracy.append(metrics['accuracy'])
        test_selected_features.append(metrics['selected_features'])
        test_selected_rate.append(metrics['selected_rate'])

        fold_index += 1
        if verbose:
            print('\n##### Finished fold {} #####\n'.format(fold_index))
            print("Test Fitness:", metrics['fitness'])
            print("Test Accuracy:", metrics['accuracy'])
            print("Selected Features:", metrics['selected_features'])
            print("Selected Rate:", metrics['selected_rate'])
            print()

    average_fitness = calculate_average_metric(metrics_each_fold, 'fitness')
    average_selected_features = calculate_average_metric(metrics_each_fold, 'selected_features')

    # Compute standard deviation of test fitness values
    std_deviation_test_fitness = np.std(test_fitness)

    # Compute mean time
    execution_time /= k

    return {
        'avg_fitness': average_fitness,
        'avg_selected_features': average_selected_features,
        'test_fitness': {
            'best': np.min(test_fitness),
            'avg': np.mean(test_fitness),
            'std_dev': std_deviation_test_fitness,
            'acc': np.mean(test_accuracy),
            'n_features': np.mean(test_selected_features),
            'selected_rate': np.mean(test_selected_rate)
        },
        'execution_time': execution_time
    }


def analysis_fitness_over_population(dataset: dict,
                                     optimizer: object,
                                     k: int = DEFAULT_FOLDS) -> float:
    """
    Analysis function to study how fitness variates over different population sizes.

    Parameters:
        - optimizer (object): The optimizer to be used.
        - dataset (dict, optional): The dataset in dict form.
        - k (int, optional): The number of folds.

    Returns:
        - average_fitness_test (float): The average fitness of the test set.
    """
    initial_population_size = 5
    max_population_size = 55
    population_size_step = 10

    key = 'generations' if 'generations' in optimizer.params else 'iterations'
    total_average_fitness_test = []
    for size in range(initial_population_size, max_population_size + 5,
                      population_size_step):
        optimizer.params[key] = size
        metrics = k_fold_cross_validation(optimizer=optimizer,
                                          dataset=dataset,
                                          k=k,
                                          verbose=False)
        total_average_fitness_test.append(metrics['test_fitness']['avg'])

    return total_average_fitness_test


def analysis_optimizers_comparison(dataset: dict,
                                   k: int = DEFAULT_FOLDS,
                                   max_iter: int = DEFAULT_ITERATIONS) -> dict:
    """
    Analysis function to compare different optimizers between each other.

    Parameters:
        - dataset (dict, optional): The dataset in dict form.
        - k (int, optional): The number of folds.
        - max_iter (int, optional): The maximum number of iterations for each optimizer.

    Returns:
        - optimizers_fitness (dict): The dictionary containing metrics of the results for each optimizer.
    """
    parameters_dict = {
        key: Optimizer.get_optimizers_parameters(key, dataset[SAMPLE].shape[1])
        for key in Optimizer.get_optimizers_names()
    }
    optimizers_fitness = {}

    for key in Optimizer.get_optimizers_names():
        fitness_values = []
        for _ in range(0, max_iter):
            metrics = k_fold_cross_validation(dataset=dataset,
                                              optimizer=Optimizer(
                                                  key, **parameters_dict[key]),
                                              k=k,
                                              verbose=False)
            fitness_values.append(np.array(metrics['test_fitness']['avg']))

        fitness_values = np.array(fitness_values)
        average_fitness = np.mean(fitness_values, axis=0)
        optimizers_fitness[key] = average_fitness

    return optimizers_fitness
