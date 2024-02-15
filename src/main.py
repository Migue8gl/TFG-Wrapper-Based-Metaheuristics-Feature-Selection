import numpy as np
from constants import *
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from data_utils import *
import time
from analysis_utils import k_fold_cross_validation, population_test, get_optimizer_parameters, optimizer_comparison
from plots import plot_fitness_over_folds, plot_fitness_over_population_sizes, plot_fitness_all_optimizers

plt.style.use(['science', 'ieee'])  # Style of plots


def main(*args, **kwargs):
    start_time = time.time()

    # Get parameters from the user
    dataset_arg = kwargs.get('-d', D2)  # Chosen dataset
    # Notifications meaning end of training
    notify_arg = kwargs.get('-n', False)
    k_arg = kwargs.get('-k', DEFAULT_FOLDS)  # Number of folds in cross validation
    scaling_arg = kwargs.get('-s', 1)  # Type of scaling applied to dataset

    # Core functionality
    dataset = load_arff_data(dataset_arg)
    if scaling_arg == 1:
        norm_dataset = scaling_min_max(dataset)
    elif scaling_arg == 2:
        norm_dataset = scaling_std_score(dataset)
    else:
        norm_dataset = dataset

    # Split the data into dict form
    dataset_dict = split_data_to_dict(norm_dataset)

    k = k_arg  # F fold cross validation
    optimizer_dict = OPTIMIZERS

    # Initial weights are set randomly between 0 and 1
    weights = np.random.uniform(low=0,
                                high=1,
                                size=dataset_dict[SAMPLE].shape[1])
    target_function_parameters = {
        'weights': weights,
        'data': dataset,
        'alpha': 0.5,
        'classifier': 'svc',
        'n_neighbors': DEFAULT_NEIGHBORS,
        'c': 0.1
    }

    for opt in optimizer_dict.keys():
        # Optimization function's parameters
        parameters, optimizer_title = get_optimizer_parameters(
            opt, dataset_dict[SAMPLE].shape[1])

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

        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        row, column = 0, 0

        # First set of plots
        second_key = list(parameters.keys())[1]
        plot_fitness_over_folds(
            fitness_values,
            parameters[second_key],
            k,
            ax=axs[row, column],
            title='Average fitness {}-fold cross validation running {} (SVC)'.
            format(k, opt))

        column += 1
        plot_fitness_over_population_sizes(
            total_fitness_test,
            np.arange(5, 60, 10),
            ax=axs[row, column],
            title='Fitness test value over population sizes for {} (SVC)'.
            format(opt))

        column = 0
        row += 1
        # Second set of plots
        plot_fitness_over_folds(
            fitness_values_2,
            parameters[second_key],
            k,
            ax=axs[row, column],
            title='Average fitness {}-fold cross validation running {} (KNN)'.
            format(k, opt))

        column += 1
        plot_fitness_over_population_sizes(
            total_fitness_test_2,
            np.arange(5, 60, 10),
            ax=axs[row, column],
            title='Fitness test value over population sizes for {} (KNN)'.
            format(opt))

        plt.tight_layout()
        plt.savefig('./images/{}'.format(opt))

    # Comparison of all optimizers
    fitness_from_all_optimizers = optimizer_comparison(
        dataset=dataset_dict,
        optimizer_dict=optimizer_dict,
        k=5,
        target_function_parameters=target_function_parameters,
        max_iterations=DEFAULT_MAX_ITERATIONS)
    # Use entire row for the last plot
    plot_fitness_all_optimizers(fitness_from_all_optimizers, 10, ax=None)
    plt.tight_layout()
    plt.savefig('./images/optimizers_comparison.jpg')

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
