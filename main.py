from pyMetaheuristic.algorithm import grasshopper_optimization_algorithm
from pyMetaheuristic.algorithm import dragonfly_algorithm
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

def plot_fitness_over_folds(fitness_values, iterations, k, ax=None, title=None):
    iteration_numbers = np.arange(0, iterations+1)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(iteration_numbers, fitness_values['TrainFitness'], label='Fitness', color='blue')
    ax.plot(iteration_numbers, fitness_values['ValFitness'], label='Validation Fitness', color='orange')
    if title == None:
        ax.set_title('Average fitness {}-fold cross validation'.format(k))
    else:
        ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness Value')
    ax.legend()

def plot_fitness_over_population_sizes(fitness_values, population_sizes, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(population_sizes, fitness_values, label='Fitness', color='purple', marker='d')
    if title == None:
        ax.set_title('Fitness test value over population sizes')
    else:
        ax.set_title(title)
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Fitness Value')
    ax.legend()

def k_fold_cross_validation(dataset, optimizator, k=5, parameters=None, target_function_parameters=None):
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
        gao, fitness_values = optimizator(target_function=fitness, target_function_parameters=target_function_parameters, **parameters)
        fitness_each_fold[fold_index] = fitness_values

        # Evaluate the model on the test set of the current fold
        target_function_parameters['data'] = sample_test
        target_function_parameters['weights'] = gao[:-2]
        test_fitness.append(fitness(**target_function_parameters)['ValFitness'])

        fold_index += 1
        print('\n##### Finished fold {} #####\n'.format(fold_index))

    # Transpose fitness values to have each list represent values for a specific index
    transposed_fitness_val = np.array([[item['ValFitness'] for item in sublist] for sublist in fitness_each_fold.values()]).T
    transposed_fitness_train = np.array([[item['TrainFitness'] for item in sublist] for sublist in fitness_each_fold.values()]).T

    # Calculate mean of each index across all lists
    average_fitness_values_train = np.mean(transposed_fitness_train, axis=1)
    average_fitness_values_val = np.mean(transposed_fitness_val, axis=1)

    return test_fitness, {'TrainFitness': average_fitness_values_train, 'ValFitness': average_fitness_values_val}

def population_test(dataset, optimizator, k=5, parameters=None, target_function_parameters=None):
    initial_population_size = 5
    max_population_size = 55
    population_size_step = 10

    total_fitness_test = []

    first_key, _ = next(iter(parameters.items()))
    total_fitness_test = [test_fitness for test_fitness, _ in (k_fold_cross_validation(dataset, optimizator, k, {first_key: size, **parameters}, target_function_parameters) 
                                                               for size in range(initial_population_size, max_population_size + 5, population_size_step))]

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

    k = 5
    optimizator = grasshopper_optimization_algorithm
    # Optimization function's parameters
    if optimizator == grasshopper_optimization_algorithm:
        parameters = {
            'grasshoppers': 20,
            'min_values': [0] * (samples.shape[1]),
            'max_values': [1] * (samples.shape[1]),
            'iterations': 650,
            'binary': 's', 
        }
    elif optimizator == dragonfly_algorithm:
        parameters = {
            'size': 20,
            'min_values': [0] * (samples.shape[1]),
            'max_values': [1] * (samples.shape[1]),
            'generations': 100,
            'binary': 's', 
        }

    # Initial weights are set randomly between 0 and 1
    weights = np.random.uniform(low=0, high=1, size=samples.shape[1])
    target_function_parameters = {
        'weights': weights,
        'data': dataset,
        'alpha': 0.5,
        'classifier': 'svc'
    }

    # SVC
    test_fitness, fitness_values = k_fold_cross_validation(dataset=dataset, optimizator=optimizator, k=k, parameters=parameters, target_function_parameters=target_function_parameters)
    total_fitness_test = population_test(dataset, optimizator, k, parameters=parameters, target_function_parameters=target_function_parameters)

    # KNN
    target_function_parameters['classifier'] = 'knn'
    test_fitness_2, fitness_values_2 = k_fold_cross_validation(dataset, optimizator, k, parameters=parameters, target_function_parameters=target_function_parameters)
    total_fitness_test_2 = population_test(dataset, optimizator, k, parameters=parameters, target_function_parameters=target_function_parameters)

    # Print average accuracy over k folds
    print('Average test fitness over 5 Folds (SVC): ', round(np.mean(test_fitness), 2))
    print('Average test fitness over 5 Folds (KNN): ', round(np.mean(test_fitness_2), 2))

    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    plot_fitness_over_folds(fitness_values, parameters['iterations'], k, ax=axes[0, 0], title='Average fitness {}-fold cross validation (SVC)'.format(k))
    plot_fitness_over_population_sizes(total_fitness_test, np.arange(5, 60, 10), ax=axes[0, 1], title='Fitness test value over population sizes (SVC)')
    plot_fitness_over_folds(fitness_values_2, parameters['iterations'], k, ax=axes[1, 0], title='Average fitness {}-fold cross validation (KNN)'.format(k))
    plot_fitness_over_population_sizes(total_fitness_test_2, np.arange(5, 60, 10), ax=axes[1, 1], title='Fitness test value over population sizes (KNN)')
    plt.tight_layout()
    plt.savefig('./images/dashboard.jpg')

    total_time = time.time() - start_time

    if(notify):
        import notifications
        notifications.send_telegram_message(message='### Ejecución Terminada - Tiempo total {} segundos ###'.format(round(total_time, 4)))
        notifications.send_telegram_image(image_path='./images/dashboard.jpg', caption='-- Dashboard de la ejecución --')

if __name__ == "__main__":
    main(notify=True)