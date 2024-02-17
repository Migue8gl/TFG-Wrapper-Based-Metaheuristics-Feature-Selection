from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
from constants import *
from data_utils import *
from optimizer import Optimizer

# TODO change names of functions
# TODO add to plot population sizes and optimizer names


def compute_accuracy(weights: np.ndarray,
                     data: dict,
                     classifier_parameters: dict,
                     classifier: str = 'knn') -> dict:
    """
    Functionality to compute the accuracy using KNN or SVC classifier

    Parameters:
        - weights (np.ndarray): The weights to be used for each feature.
        - data (dict): The dataset in dict form splitted into samples and labels.
        - classifier_parameters (dict): The classifier parameters. 
        - classifier (str, optional): The classifier to be used.

    Returns:
        - errors (dict): The dictionary containing e_in error and e_out error.
    """
    sample = data[SAMPLE]
    labels = data[LABELS]

    # Giving each characteristic an importance by multiplying the sample and weights
    sample_weighted = np.multiply(sample, weights)
    # Split into train and test data
    x_train, x_test, y_train, y_test = train_test_split(sample_weighted,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42)

    if (classifier == 'knn'):
        classifier = KNeighborsClassifier(
            n_neighbors=classifier_parameters['n_neighbors'],
            weights=classifier_parameters['weights'])
    elif (classifier == 'svc'):
        classifier = SVC(C=classifier_parameters['C'],
                         kernel=classifier_parameters['kernel'])
    else:
        print('No valid classifier, using KNN by default')
        classifier = KNeighborsClassifier(
            n_neighbors=classifier_parameters['n_neighbors'],
            weights=classifier_parameters['weights'])

    # Train the classifier
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_train)
    e_in = accuracy_score(y_train, y_pred)

    y_pred = classifier.predict(x_test)
    e_out = accuracy_score(y_test, y_pred)

    return {'TrainError': e_in, 'ValError': e_out}


def fitness(weights: np.ndarray,
            data: dict,
            classifier_parameters: dict,
            alpha: float = 0.5,
            classifier: str = 'knn') -> dict:
    """
    Functionality to compute the accuracy using KNN or SVC classifier

    Parameters:
        - weights (np.ndarray): The weights to be used for each feature.
        - data (dict): The dataset in dict form splitted into samples and labels.
        - classifier_parameters (dict): The classifier parameters. 
        - classifier (str, optional): The classifier to be used.
        
    Returns:
        - fitness (dict): The dictionary containing training fitness and validation fitness.
    """
    # Count number of features with zero importance.
    reduction_count = np.sum(weights == 0)
    classification_rate = compute_accuracy(
        weights,
        data=data,
        classifier=classifier,
        classifier_parameters=classifier_parameters)
    reduction_rate = reduction_count / len(weights)

    # Calculate the error rates in training
    classification_error = 1 - classification_rate['TrainError']
    reduction_error = 1 - reduction_rate

    # Compute fitness as a combination of classification and reduction errors
    fitness_train = alpha * classification_error + \
        (1 - alpha) * reduction_error
    classification_error = 1 - classification_rate['ValError']
    fitness_val = alpha * classification_error + (1 - alpha) * reduction_error

    return {'TrainFitness': fitness_train, 'ValFitness': fitness_val}


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
                            k: int = 5) -> dict:
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
        test_fitness.append(
            fitness(**optimizer.params['target_function_parameters'])
            ['ValFitness'])

        fold_index += 1
        print('\n##### Finished fold {} #####\n'.format(fold_index))

    average_fitness_val = calculate_average_fitness(fitness_each_fold,
                                                    'ValFitness')
    average_fitness_train = calculate_average_fitness(fitness_each_fold,
                                                      'TrainFitness')
    # Compute standard deviation of test fitness values
    std_deviation_test_fitness = np.std(test_fitness)

    return {
        'TrainFitness': average_fitness_train,
        'ValFitness': average_fitness_val,
        'TestFitness': {
            'Best': np.max(test_fitness),
            'Average': np.mean(test_fitness),
            'StandardDeviation': std_deviation_test_fitness
        }
    }


def population_test(dataset: dict, optimizer: object, k: int = 5) -> float:
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

    key = 'generation' if 'generation' in optimizer.params else 'iteration'
    total_average_fitness_test = []
    for size in range(initial_population_size, max_population_size + 5,
                      population_size_step):
        optimizer.params[key] = size
        metrics = k_fold_cross_validation(optimizer=optimizer,
                                          dataset=dataset,
                                          k=k)
        total_average_fitness_test.append(metrics['TestFitness']['Average'])

    average_fitness_test = np.mean(total_average_fitness_test)

    return average_fitness_test


def optimizer_comparison(dataset: dict, k: int = 5) -> dict:
    """
    Analysis function to compare different optimizers between each other.

    Parameters:
        - dataset (dict, optional): The dataset in dict form.
        - k (int, optional): The number of folds.

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
        for _ in range(0, DEFAULT_MAX_ITERATIONS):
            metrics = k_fold_cross_validation(dataset=dataset,
                                              optimizer=Optimizer(
                                                  key, **parameters_dict[key]),
                                              k=k)
            fitness_values.append(np.array(metrics['TestFitness']['Average']))

        fitness_values = np.array(fitness_values)
        average_fitness = np.mean(fitness_values, axis=0)
        optimizers_fitness[key] = average_fitness

    return optimizers_fitness
