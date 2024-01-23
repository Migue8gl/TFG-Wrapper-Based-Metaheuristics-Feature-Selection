import numpy as np
from constants import *
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from data_utils import *
import time
from test_utils import *
from analysis_utils import k_fold_cross_validation, population_test, get_optimizer_parameters, optimizer_comparison
from plots import plot_fitness_over_folds, plot_fitness_over_population_sizes, plot_fitness_all_optimizers


def main(*args, **kwargs):
    start_time = time.time()

    # Get parameters from the user
    dataset_arg = kwargs.get('-d', D2)  # Chosen dataset
    # Notifications meaning end of training
    notify_arg = kwargs.get('-n', True)
    k_arg = kwargs.get('-k', 5)  # Number of folds in cross validation
    scaling_arg = kwargs.get('-s', 1)  # Type of scaling applied to dataset

    # Core functionality
    dataset = load_arff_data(dataset_arg)
    if scaling_arg == 1:
        norm_dataset = scaling_min_max(dataset)
    elif scaling_arg == 2:
        norm_dataset = scaling_standard_score(dataset)
    else:
        norm_dataset = dataset

    # Split the data into dict form
    dataset_dict = split_data_to_dict(norm_dataset)

    k = k_arg  # F fold cross validation
    optimizer_dict = OPTIMIZERS

    # Initial weights are set randomly between 0 and 1
    weights = np.random.uniform(low=0,
                                high=1,
                                size=dataset_dict[DATA].shape[1])
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
        parameters, optimizer_title = get_optimizer_parameters(
            opt, dataset_dict[DATA].shape[1])

        # SVC Cross validation for x optimizer alone
        test_fitness, fitness_values = k_fold_cross_validation(
            dataset=dataset_dict,
            optimizer=optimizer_dict[opt],
            k=k,
            parameters=parameters,
            target_function_parameters=target_function_parameters)

        # Test for x optimizer altering population size with SVC
        total_fitness_test = population_test(
            dataset=dataset_dict,
            optimizer=optimizer_dict[opt],
            k=k,
            parameters=parameters,
            target_function_parameters=target_function_parameters)

        # KNN cross validation for x optimizer alone
        target_function_parameters['classifier'] = 'knn'
        test_fitness_2, fitness_values_2 = k_fold_cross_validation(
            dataset=dataset_dict,
            optimizer=optimizer_dict[opt],
            k=k,
            parameters=parameters,
            target_function_parameters=target_function_parameters)

        # Test for x optimizer altering population size with KNN
        total_fitness_test_2 = population_test(
            dataset=dataset_dict,
            optimizer=optimizer_dict[opt],
            k=k,
            parameters=parameters,
            target_function_parameters=target_function_parameters)

        # Print average accuracy over k folds
        print('Average test fitness over {} Folds (SVC): {}'.format(
            k, round(np.mean(test_fitness), 2)))
        print('Average test fitness over {} Folds (KNN): {}'.format(
            k, round(np.mean(test_fitness_2), 2)))

        # First set of plots
        second_key = list(parameters.keys())[1]
        ax1 = fig.add_subplot(gs[rows, columns])
        plot_fitness_over_folds(
            fitness_values,
            parameters[second_key],
            k,
            ax=ax1,
            title='Average fitness {}-fold cross validation (SVC)'.format(k))
        columns += 1
        ax2 = fig.add_subplot(gs[rows, columns])
        plot_fitness_over_population_sizes(
            total_fitness_test,
            np.arange(5, 60, 10),
            ax=ax2,
            title='Fitness test value over population sizes (SVC)')

        # Second set of plots
        rows += 1
        columns = 0
        ax3 = fig.add_subplot(gs[rows, columns])
        plot_fitness_over_folds(
            fitness_values_2,
            parameters[second_key],
            k,
            ax=ax3,
            title='Average fitness {}-fold cross validation (KNN)'.format(k))
        columns += 1
        ax4 = fig.add_subplot(gs[rows, columns])
        plot_fitness_over_population_sizes(
            total_fitness_test_2,
            np.arange(5, 60, 10),
            ax=ax4,
            title='Fitness test value over population sizes (KNN)')
        rows += 1

    # Third set of plots
    ax5 = fig.add_subplot(gs[rows, :])

    # Comparison of all optimizers
    fitness_from_all_optimizers = optimizer_comparison(
        dataset=dataset_dict,
        optimizer_dict=optimizer_dict,
        k=5,
        target_function_parameters=target_function_parameters,
        max_iterations=DEFAULT_MAX_ITERATIONS)
    # Use entire row for the last plot
    plot_fitness_all_optimizers(fitness_from_all_optimizers, 10, ax=ax5)

    fig.suptitle(
        PLOT_TITLE, fontsize=16
    )  # TODO fix title for all plots and fix subtitle for subplots
    plt.tight_layout()
    plt.savefig('./images/dashboard.jpg')

    total_time = time.time() - start_time

    if (notify_arg):
        import notifications
        notifications.send_telegram_message(
            message='### Ejecución Terminada - Tiempo total {} segundos ###'.
            format(round(total_time, 4)))
        notifications.send_telegram_image(
            image_path='./images/dashboard.jpg',
            caption='-- Dashboard de la ejecución --')


if __name__ == "__main__":
    main()
