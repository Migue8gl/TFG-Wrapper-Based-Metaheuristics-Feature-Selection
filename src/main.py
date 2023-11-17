from pyMetaheuristic.algorithm import grasshopper_optimization_algorithm
from pyMetaheuristic.algorithm import dragonfly_algorithm
import numpy as np
import arff
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
from src.analysis_utils import k_fold_cross_validation, population_test
from plots import plot_fitness_over_folds, plot_fitness_over_population_sizes

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

    k = 5 # F fold cross validation
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
            'size': 30,
            'min_values': [0] * (samples.shape[1]),
            'max_values': [1] * (samples.shape[1]),
            'generations': 500,
            'binary': 's', 
        }

    # Initial weights are set randomly between 0 and 1
    weights = np.random.uniform(low=0, high=1, size=samples.shape[1])
    target_function_parameters = {
        'weights': weights,
        'data': dataset,
        'alpha': 0.5,
        'classifier': 'svc',
        'n_neighbors': 20
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
    fig.suptitle('Running DA', fontsize=16)
    plt.tight_layout()
    plt.savefig('./images/dashboard.jpg')

    total_time = time.time() - start_time

    if(notify):
        import notifications
        notifications.send_telegram_message(message='### Ejecución Terminada - Tiempo total {} segundos ###'.format(round(total_time, 4)))
        notifications.send_telegram_image(image_path='./images/dashboard.jpg', caption='-- Dashboard de la ejecución --')

if __name__ == "__main__":
    try:
        main(notify=True)
    except Exception as e:
        import notifications
        notifications.send_telegram_message('### Ha ocurrido un error en la ejecución del programa ###')
        notifications.send_telegram_message('Error: {}'.format(e))
