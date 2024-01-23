from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
from constants import *
from data_utils import *

# TODO change names of functions
# TODO add to plot population sizes and optimizer names


def compute_accuracy(weights, data, classifier='knn', n_neighbors=5):
    sample = data[DATA]
    labels = data[LABELS]

    sample_weighted = np.multiply(sample, weights)
    x_train, x_test, y_train, y_test = train_test_split(sample_weighted,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42)

    if (classifier == 'knn'):
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                          weights='distance')
    elif (classifier == 'svc'):
        classifier = SVC(kernel='rbf')
    else:
        print('No valid classifier, using KNN by default')
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                          weights='distance')

    # Train the classifier
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_train)
    e_in = accuracy_score(y_train, y_pred)

    y_pred = classifier.predict(x_test)
    e_out = accuracy_score(y_test, y_pred)

    return {'TrainError': e_in, 'ValError': e_out}


def fitness(weights, data, alpha=0.5, classifier='knn', n_neighbors=5):
    reduction_count = np.sum(weights == 0)
    weights[weights < 0.1] = 0.0
    classification_rate = compute_accuracy(weights,
                                           data=data,
                                           classifier=classifier,
                                           n_neighbors=n_neighbors)
    reduction_rate = reduction_count / len(weights)

    # Calculate the error as a percentage
    classification_error = 1 - classification_rate['TrainError']
    reduction_error = 1 - reduction_rate

    # Compute fitness as a combination of classification and reduction errors
    fitness_train = alpha * classification_error + \
        (1 - alpha) * reduction_error
    classification_error = 1 - classification_rate['ValError']
    fitness_val = alpha * classification_error + (1 - alpha) * reduction_error

    return {'TrainFitness': fitness_train, 'ValFitness': fitness_val}


def calculate_average_fitness(fitness_each_fold, fitness_key):
    """
    Calculate the average fitness values from a dictionary of fitness values for each fold.

    Parameters:
    - fitness_each_fold: Dictionary of fitness values for each fold.
    - fitness_key: Key specifying the fitness values to extract ('ValFitness' or 'TrainFitness').

    Returns:
    - An array of average fitness values across all folds.
    """
    # Transpose fitness values to have each list represent values for a specific index
    transposed_fitness = np.array([[item[fitness_key] for item in sublist]
                                   for sublist in fitness_each_fold.values()
                                   ]).T

    # Calculate mean of each index across all lists
    average_fitness_values = np.mean(transposed_fitness, axis=1)

    return average_fitness_values


def k_fold_cross_validation(dataset,
                            optimizer,
                            k=5,
                            parameters=None,
                            target_function_parameters=None):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    test_fitness = []
    fitness_each_fold = {}
    fold_index = 0

    for train_index, test_index in skf.split(dataset[DATA], dataset[LABELS]):
        x_train, x_test = dataset[DATA][train_index], dataset[DATA][test_index]
        y_train, y_test = dataset[LABELS][train_index], dataset[LABELS][
            test_index]

        sample = {DATA: x_train, LABELS: y_train}
        sample_test = {DATA: x_test, LABELS: y_test}

        # Override the data to be optimized in the search process
        target_function_parameters[DATA] = sample

        # Run optimization algorithm on the current fold
        result, fitness_values = optimizer(
            target_function=fitness,
            target_function_parameters=target_function_parameters,
            **parameters)
        fitness_each_fold[fold_index] = fitness_values

        # Evaluate the model on the test set of the current fold
        target_function_parameters[DATA] = sample_test
        target_function_parameters['weights'] = result[:-2]
        test_fitness.append(
            fitness(**target_function_parameters)['ValFitness'])

        fold_index += 1
        print('\n##### Finished fold {} #####\n'.format(fold_index))

    average_fitness_val = calculate_average_fitness(fitness_each_fold,
                                                    'ValFitness')
    average_fitness_train = calculate_average_fitness(fitness_each_fold,
                                                      'TrainFitness')

    return test_fitness, {
        'TrainFitness': average_fitness_train,
        'ValFitness': average_fitness_val
    }


def population_test(dataset,
                    optimizer,
                    k=5,
                    parameters=None,
                    target_function_parameters=None):
    initial_population_size = 5
    max_population_size = 55
    population_size_step = 10

    total_fitness_test = []

    first_key = next(iter(parameters.keys()))
    total_fitness_test = [
        test_fitness for test_fitness, _ in (
            k_fold_cross_validation(dataset, optimizer, k, {
                first_key: size,
                **parameters
            }, target_function_parameters)
            for size in range(initial_population_size, max_population_size +
                              5, population_size_step))
    ]

    total_fitness_array = np.array(total_fitness_test).T
    average_fitness_test = np.mean(total_fitness_array, axis=0)

    return average_fitness_test


def optimizer_comparison(dataset,
                         optimizer_dict,
                         k=5,
                         target_function_parameters=None,
                         max_iterations=30):
    parameters_dict = {
        key: get_optimizer_parameters(key, dataset[DATA].shape[1])
        for key in optimizer_dict.keys()
    }
    optimizers_fitness = {}

    for key in optimizer_dict.keys():
        fitness_values = []
        for _ in range(0, max_iterations):
            _, fitness_val = k_fold_cross_validation(
                dataset=dataset,
                optimizer=optimizer_dict[key],
                k=k,
                parameters=parameters_dict[key][0],
                target_function_parameters=target_function_parameters)
            fitness_values.append(np.array(fitness_val['ValFitness']))

        fitness_values = np.array(fitness_values)
        average_fitness = np.mean(fitness_values, axis=0)
        optimizers_fitness[key] = average_fitness

    return optimizers_fitness
