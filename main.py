from pyMetaheuristic.algorithm import grasshopper_optimization_algorithm
import numpy as np
import arff
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import time

def load_arff_data(file_path):
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

def normalize_data(data):
    # Separate the features (x) and labels (y)
    x = data[:, :-1].astype(np.float64)
    y = data[:, -1]

    scaler = MinMaxScaler()

    # Fit the scaler to the features and normalize the data between 0 and 1
    x_normalized = scaler.fit_transform(x)

    # Combine the normalized features (x_normalized) and labels (y) into a single array
    normalized_data = np.column_stack((x_normalized, y))

    return normalized_data

def compute_accuracy(weights, data, classifier='knn', number_neighbors=5):
    sample = data['data']
    labels = data['labels']

    sample_weighted = np.multiply(sample, weights)
    x_train, x_test, y_train, y_test = train_test_split(sample_weighted, labels, test_size=0.2, random_state=42)

    if (classifier == 'knn'):
        classifier = KNeighborsClassifier(n_neighbors=number_neighbors, weights='distance')
    elif (classifier == 'svc'):
        classifier = SVC(kernel='rbf')
    else:
        print('No valid classifier, using KNN by default')
        classifier = KNeighborsClassifier(n_neighbors=number_neighbors, weights='distance')

    # Train the classifier
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_train)
    e_in = accuracy_score(y_train, y_pred)

    y_pred = classifier.predict(x_test)
    e_out = accuracy_score(y_test, y_pred)

    return {'TrainError': e_in, 'ValError': e_out}

def fitness(weights, data, alpha=0.5, classifier='knn'):
    reduction_count = np.sum(weights == 0)
    weights[weights < 0.1] = 0.0
    classification_rate = compute_accuracy(weights, data=data, classifier=classifier)
    reduction_rate = reduction_count / len(weights)

    # Calculate the error as a percentage
    classification_error = 1 - classification_rate['TrainError']
    reduction_error = 1 - reduction_rate

    # Compute fitness as a combination of classification and reduction errors
    fitness_train = alpha * classification_error + (1 - alpha) * reduction_error
    classification_error = 1 - classification_rate['ValError']
    fitness_val = alpha * classification_error + (1 - alpha) * reduction_error

    return {'TrainFitness': fitness_train, 'ValFitness': fitness_val}

def plot_fitness_over_folds(fitness_values, iterations, k, ax=None):
    iteration_numbers = np.arange(0, iterations+1)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(iteration_numbers, fitness_values['TrainFitness'], label='Fitness', color='blue')
    ax.plot(iteration_numbers, fitness_values['ValFitness'], label='Validation Fitness', color='orange')
    ax.set_title('Average fitness {}-fold cross validation'.format(k))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness Value')
    ax.legend()

def plot_fitness_over_population_sizes(fitness_values, population_sizes, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(population_sizes, fitness_values, label='Fitness', color='purple', marker='d')
    ax.set_title('Fitness test value over population sizes')
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Fitness Value')
    ax.legend()

def k_fold_cross_validation(dataset, k=5, parameters=None, target_function_parameters=None):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    test_fitness = []
    fitness_each_fold = {}
    fold_index = 0

    for train_index, test_index in skf.split(dataset['data'], dataset['labels']):
        x_train, x_test = dataset['data'][train_index], dataset['data'][test_index]
        y_train, y_test = dataset['labels'][train_index], dataset['labels'][test_index]

        sample = {'data': x_train, 'labels': y_train}
        sample_test = {'data': x_test, 'labels': y_test}

        # Override the data to be optimized in the search process
        target_function_parameters['data'] = sample

        # Run optimization algorithm on the current fold
        gao, fitness_values = grasshopper_optimization_algorithm(target_function=fitness, target_function_parameters=target_function_parameters, **parameters)
        fitness_each_fold[fold_index] = fitness_values

        # Evaluate the model on the test set of the current fold
        target_function_parameters['data'] = sample_test
        target_function_parameters['weights'] = gao[:-1]
        test_fitness.append(fitness(**target_function_parameters)['ValFitness'])

        fold_index += 1
        print('\n##### Finished {} fold #####\n'.format(fold_index))

    # Transpose fitness values to have each list represent values for a specific index
    transposed_fitness_val = np.array([[item['ValFitness'] for item in sublist] for sublist in fitness_each_fold.values()]).T
    transposed_fitness_train = np.array([[item['TrainFitness'] for item in sublist] for sublist in fitness_each_fold.values()]).T

    # Calculate mean of each index across all lists
    average_fitness_values_train = np.mean(transposed_fitness_train, axis=1)
    average_fitness_values_val = np.mean(transposed_fitness_val, axis=1)

    return test_fitness, {'TrainFitness': average_fitness_values_train, 'ValFitness': average_fitness_values_val}

def population_test(dataset, k=5, parameters=None, target_function_parameters=None):
    initial_population_size = 5
    max_population_size = 50
    population_size_step = 5

    total_fitness_test = []

    for size in range(initial_population_size, max_population_size + 5, population_size_step):
        parameters['grasshoppers'] = size
        test_fitness, _ = k_fold_cross_validation(dataset, k, parameters, target_function_parameters)
        total_fitness_test.append(test_fitness)

    total_fitness_array = np.array(total_fitness_test).T
    average_fitness_test = np.mean(total_fitness_array, axis=0)

    return average_fitness_test

def main(notify=False):
    start_time = time.time()

    # Read data
    d1 = './datasets/spectf-heart.arff'
    d2 = './datasets/ionosphere.arff'
    d3 = './datasets/parkinsons.arff'
    dataset = load_arff_data(d2)

    # Split the data into training and testing sets
    samples = dataset[:, :-1].astype(np.float64)
    classes = dataset[:, -1]
    dataset = {'data': samples, 'labels': classes}

    # Optimization function's parameters
    parameters = {
        'grasshoppers': 20,
        'min_values': [0] * (samples.shape[1]),
        'max_values': [1] * (samples.shape[1]),
        'iterations': 650,
        'binary': 's', 
    }

    # Initial weights are set randomly between 0 and 1
    weights = np.random.uniform(low=0, high=1, size=samples.shape[1])
    target_function_parameters = {
        'weights': weights,
        'data': dataset,
        'alpha': 0.5,
        'classifier': 'knn'
    }

    # Perform k-fold cross-validation
    k = 5
    test_fitness, fitness_values = k_fold_cross_validation(dataset, k, parameters=parameters, target_function_parameters=target_function_parameters)
    total_fitness_test = population_test(dataset, k, parameters=parameters, target_function_parameters=target_function_parameters)

    # Print average accuracy over k folds
    print('Average test fitness over 5 Folds: ', round(np.mean(test_fitness), 2))

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 5))
    plot_fitness_over_folds(fitness_values, parameters['iterations'], k, ax=axes[0])
    plot_fitness_over_population_sizes(total_fitness_test, np.arange(5, 55, 5), ax=axes[1])
    plt.tight_layout()
    plt.savefig('./images/dashboard.jpg')

    total_time = time.time() - start_time

    if(notify):
        import notifications
        notifications.send_telegram_message(message='### Ejecución Terminada - Tiempo total {} segundos ###'.format(round(total_time, 4)))
        notifications.send_telegram_image(image_path='./images/dashboard.jpg', caption='-- Dashboard de la ejecución --')

if __name__ == "__main__":
    main(notify=True)