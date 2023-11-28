from pyMetaheuristic.algorithm import grasshopper_optimization_algorithm
from pyMetaheuristic.algorithm import dragonfly_algorithm
from pyMetaheuristic.algorithm import grey_wolf_optimizer
from pyMetaheuristic.algorithm import whale_optimization_algorithm
import numpy as np
from matplotlib.gridspec import GridSpec
import arff
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
from analysis_utils import k_fold_cross_validation, population_test, get_optimizer_parameters, optimizator_comparison
from plots import plot_fitness_over_folds, plot_fitness_over_population_sizes, plot_fitness_all_optimizers

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
    dataset = normalize_data(load_arff_data(d2))

    # Split the data into training and testing sets
    samples = dataset[:, :-1].astype(np.float64)
    classes = dataset[:, -1]
    dataset = {'data': samples, 'labels': classes}

    k = 5 # F fold cross validation
    optimizer_dict = {'WOA': whale_optimization_algorithm}

    # Initial weights are set randomly between 0 and 1
    weights = np.random.uniform(low=0, high=1, size=samples.shape[1])
    target_function_parameters = {
        'weights': weights,
        'data': dataset,
        'alpha': 0.5,
        'classifier': 'svc',
        'n_neighbors': 20
    }
    
    fig = plt.figure(figsize=(10, 30))
    gs = GridSpec(2 * len(optimizer_dict) + 1, 2, figure=fig)
    rows, columns = 0, 0

    for opt in optimizer_dict.keys():
        columns = 0
        # Optimization function's parameters
        parameters, optimizer_title = get_optimizer_parameters(opt, samples.shape[1])

        # SVC
        test_fitness, fitness_values = k_fold_cross_validation(dataset=dataset, optimizator=optimizer_dict[opt], k=k, parameters=parameters, target_function_parameters=target_function_parameters)
        total_fitness_test = population_test(dataset=dataset, optimizator=optimizer_dict[opt], k=k, parameters=parameters, target_function_parameters=target_function_parameters)

        # KNN
        target_function_parameters['classifier'] = 'knn'
        test_fitness_2, fitness_values_2 = k_fold_cross_validation(dataset=dataset, optimizator=optimizer_dict[opt], k=k, parameters=parameters, target_function_parameters=target_function_parameters)
        total_fitness_test_2 = population_test(dataset=dataset, optimizator=optimizer_dict[opt], k=k, parameters=parameters, target_function_parameters=target_function_parameters)

        # Print average accuracy over k folds
        print('Average test fitness over 5 Folds (SVC): ', round(np.mean(test_fitness), 2))
        print('Average test fitness over 5 Folds (KNN): ', round(np.mean(test_fitness_2), 2))

        # First set of plots
        second_key = list(parameters.keys())[1]
        ax1 = fig.add_subplot(gs[rows, columns])
        plot_fitness_over_folds(fitness_values, parameters[second_key], k, ax=ax1, title='Average fitness {}-fold cross validation (SVC)'.format(k))
        columns += 1
        ax2 = fig.add_subplot(gs[rows, columns])
        plot_fitness_over_population_sizes(total_fitness_test, np.arange(5, 60, 10), ax=ax2, title='Fitness test value over population sizes (SVC)')

        # Second set of plots
        rows += 1
        columns = 0
        ax3 = fig.add_subplot(gs[rows, columns])
        plot_fitness_over_folds(fitness_values_2, parameters[second_key], k, ax=ax3, title='Average fitness {}-fold cross validation (KNN)'.format(k))
        columns += 1
        ax4 = fig.add_subplot(gs[rows, columns])
        plot_fitness_over_population_sizes(total_fitness_test_2, np.arange(5, 60, 10), ax=ax4, title='Fitness test value over population sizes (KNN)')
        rows += 1

    # Third set of plots
    ax5 = fig.add_subplot(gs[rows, :])
    max_iterations = 30
    fitness_from_all_optimizers = optimizator_comparison(dataset=dataset, optimizer_dict=optimizer_dict, k=5, target_function_parameters=target_function_parameters, max_iterations=max_iterations)
    plot_fitness_all_optimizers(fitness_from_all_optimizers, 10, ax=ax5)  # Use entire row for the last plot

    fig.suptitle(optimizer_title, fontsize=16)
    plt.tight_layout()
    plt.savefig('./images/dashboard.jpg')

    total_time = time.time() - start_time

    if(notify):
        import notifications
        notifications.send_telegram_message(message='### Ejecución Terminada - Tiempo total {} segundos ###'.format(round(total_time, 4)))
        notifications.send_telegram_image(image_path='./images/dashboard.jpg', caption='-- Dashboard de la ejecución --')

if __name__ == "__main__":
    main(notify=True)
   