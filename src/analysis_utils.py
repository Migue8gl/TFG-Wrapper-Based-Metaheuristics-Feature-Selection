import time
from typing import Optional

import numpy as np
from constants import DATA, DEFAULT_FOLDS, DEFAULT_ITERATIONS, LABELS, SAMPLE
from optimizer import Optimizer
from sklearn.model_selection import StratifiedKFold


def calculate_average_fitness(fitness_each_fold: dict,
                              fitness_key: str) -> list:
    """
    Calculate the average fitness values from a dictionary of fitness values for each fold.

    Parameters:
        - fitness_each_fold (dict): Dictionary of fitness values for each fold.
        - fitness_key (str): Key specifying the fitness values to extract ('ValFitness' or 'TrainFitness').

    Returns:
        - average_fitness_values (list): An array of average fitness values across all folds.
    """
    # Transpose fitness values to have each list represent values for a specific index
    transposed_fitness = np.array([[item[fitness_key] for item in sublist]
                                   for sublist in fitness_each_fold.values()
                                   ]).T

    # Calculate mean of each index across all lists
    average_fitness_values = np.mean(transposed_fitness, axis=1)

    return average_fitness_values


def k_fold_cross_validation(optimizer: object,
                            dataset: Optional[dict] = None,
                            k: int = DEFAULT_FOLDS,
                            verbose: bool = False) -> dict:
    """
    Implementation of k-fold cross-validation.

    Parameters:
        - optimizer (object): The optimizer to be used.
        - dataset (dict, optional): The dataset in dict form.
        - k (int, optional): The number of folds.

    Returns:
        - metrics (dict): The dictionary containing metrics of the results.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    test_fitness = []
    fitness_each_fold = {}
    fold_index = 0

    for train_index, test_index in skf.split(dataset[SAMPLE], dataset[LABELS]):
        x_train, x_test = dataset[SAMPLE][train_index], dataset[SAMPLE][
            test_index]
        y_train, y_test = dataset[LABELS][train_index], dataset[LABELS][
            test_index]

        sample = {SAMPLE: x_train, LABELS: y_train}
        sample_test = {SAMPLE: x_test, LABELS: y_test}

        # Run optimization algorithm on the current fold
        result, fitness_values = optimizer.optimize(sample)
        fitness_each_fold[fold_index] = fitness_values

        # Evaluate the model on the test set of the current fold
        optimizer.params['target_function_parameters'][DATA] = sample_test
        optimizer.params['target_function_parameters']['weights'] = result[:-2]
        start_time = time.time()
        metrics = Optimizer.fitness(
            **optimizer.params['target_function_parameters'])
        end_time = time.time()
        execution_time = end_time - start_time
        test_fitness.append(metrics['validation']['fitness'])

        fold_index += 1
        if verbose:
            print('\n##### Finished fold {} #####\n'.format(fold_index))

    average_fitness_val = calculate_average_fitness(fitness_each_fold,
                                                    'val_fitness')
    average_fitness_train = calculate_average_fitness(fitness_each_fold,
                                                      'train_fitness')
    # Compute standard deviation of test fitness values
    std_deviation_test_fitness = np.std(test_fitness)

    return {
        'avg_train_fitness': average_fitness_train,
        'avg_val_fitness': average_fitness_val,
        'test_fitness': {
            'best': np.min(test_fitness),
            'avg': np.mean(test_fitness),
            'std_dev': std_deviation_test_fitness,
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
