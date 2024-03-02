import time

import matplotlib.pyplot as plt
import notifications
import numpy as np
from analysis_utils import (
    analysis_optimizers_comparison,
    analysis_fitness_over_population,
    k_fold_cross_validation,
)
from constants import (
    D2,
    DEFAULT_FOLDS,
    DEFAULT_ITERATIONS,
    DEFAULT_LOWER_BOUND,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_NEIGHBORS,
    DEFAULT_UPPER_BOUND,
    SAMPLE,
    SVC_CLASSIFIER,
)
from data_utils import (
    load_arff_data,
    scaling_min_max,
    scaling_std_score,
    split_data_to_dict,
)
from optimizer import Optimizer
from plots import (
    plot_fitness_all_optimizers,
    plot_fitness_over_folds,
    plot_fitness_over_population_sizes,
)

plt.style.use(['science', 'ieee'])  # Style of plots


def main(*args, **kwargs):
    start_time = time.time()

    # Get parameters from the user
    dataset_arg = kwargs.get('-d', D2)  # Chosen dataset
    # Notifications meaning end of training
    notify_arg = kwargs.get('-n', True)
    k_arg = kwargs.get('-k',
                       DEFAULT_FOLDS)  # Number of folds in cross validation
    scaling_arg = kwargs.get('-s', 3)  # Type of scaling applied to dataset

    # Core functionality
    dataset = load_arff_data(dataset_arg)
    if scaling_arg == 1:
        norm_dataset = scaling_min_max(dataset)
    elif scaling_arg == 2:
        norm_dataset = scaling_std_score(dataset)
    elif scaling_arg == 3:
        norm_dataset = scaling_min_max(scaling_std_score(dataset))
    else:
        norm_dataset = dataset

    # Split the data into dict form
    dataset_dict = split_data_to_dict(norm_dataset)

    k = k_arg  # F fold cross validation

    # Initial target function's parameters
    target_function_parameters = {
        'weights':
        np.random.uniform(low=DEFAULT_LOWER_BOUND,
                          high=DEFAULT_UPPER_BOUND,
                          size=dataset_dict[SAMPLE].shape[1]),
        'data':
        None,
        'alpha':
        0.5,
        'classifier':
        SVC_CLASSIFIER,
        'classifier_parameters': {
            'n_neighbors': DEFAULT_NEIGHBORS,
            'weights': 'distance',
            'C': 1,
            'kernel': 'rbf'
        }
    }

    total_iterations = len(Optimizer.get_optimizers_names()) * 2
    current_iteration = 0

    for opt in Optimizer.get_optimizers_names():
        current_iteration += 1
        progress_percentage = (current_iteration / total_iterations) * 100
        print(f"Progress: {progress_percentage:.2f}%")

        # Optimization function's parameters
        parameters = Optimizer.get_default_optimizer_parameters(
            opt.upper(), dataset_dict[SAMPLE].shape[1])
        parameters['target_function_parameters'] = target_function_parameters

        # Creating optimizer object
        optimizer = Optimizer(opt, parameters)

        # SVC Cross validation for x optimizer alone
        metrics = k_fold_cross_validation(dataset=dataset_dict,
                                          optimizer=optimizer,
                                          k=k)

        # Test for optimizer X altering population size with SVC
        fitness_over_different_populations = analysis_fitness_over_population(
            dataset=dataset_dict, optimizer=optimizer, k=k)

        # KNN cross validation for optimizer X alone
        optimizer.params['target_function_parameters']['classifier'] = 'knn'
        metrics2 = k_fold_cross_validation(dataset=dataset_dict,
                                           optimizer=optimizer,
                                           k=k)

        # Test for optimizer X altering population size with KNN
        fitness_over_different_populations2 = analysis_fitness_over_population(
            dataset=dataset_dict, optimizer=optimizer, k=k)

        # Print average accuracy over k folds
        print('Average test fitness over {} Folds (SVC): {}'.format(
            k, round(metrics['TestFitness']['Average'], 2)))
        print('Average test fitness over {} Folds (KNN): {}'.format(
            k, round(metrics2['TestFitness']['Average'], 2)))

        _, axs = plt.subplots(2, 2, figsize=(10, 5))
        row, column = 0, 0

        # First set of plots
        second_key = list(parameters.keys())[1]
        plot_fitness_over_folds(
            metrics,
            parameters[second_key],
            k,
            ax=axs[row, column],
            title='Average fitness {}-fold cross validation running {} (SVC)'.
            format(k, opt))

        column += 1
        plot_fitness_over_population_sizes(
            fitness_over_different_populations,
            np.arange(5, 60, 10),
            ax=axs[row, column],
            title='Fitness test value over population sizes for {} (SVC)'.
            format(opt))

        column = 0
        row += 1
        # Second set of plots
        plot_fitness_over_folds(
            metrics2,
            parameters[second_key],
            k,
            ax=axs[row, column],
            title='Average fitness {}-fold cross validation running {} (KNN)'.
            format(k, opt))

        column += 1
        plot_fitness_over_population_sizes(
            fitness_over_different_populations2,
            np.arange(5, 60, 10),
            ax=axs[row, column],
            title='Fitness test value over population sizes for {} (KNN)'.
            format(opt))

        plt.tight_layout()
        plt.savefig('./images/{}'.format(opt))

        # Send progress message to Telegram
        if (notify_arg):
            token, chat_id = notifications.load_credentials(
                './credentials/credentials.txt')
            notifications.send_telegram_message(
                token=token,
                chat_id=chat_id,
                message=f'Progress: {progress_percentage:.2f}%')

    # Comparison of all optimizers
    fitness_from_all_optimizers = analysis_optimizers_comparison(
        dataset=dataset_dict, k=DEFAULT_FOLDS, max_iter=DEFAULT_MAX_ITERATIONS)

    plot_fitness_all_optimizers(fitness_from_all_optimizers,
                                DEFAULT_ITERATIONS,
                                ax=None)
    plt.tight_layout()
    plt.savefig('./images/optimizers_comparison.jpg')

    total_time = time.time() - start_time

    if (notify_arg):
        token, chat_id = notifications.load_credentials(
            './credentials/credentials.txt')

        notifications.send_telegram_message(
            token=token,
            chat_id=chat_id,
            message='### Execution finished - Total time {} seconds ###'.
            format(round(total_time, 4)))

        for opt_name in Optimizer.get_optimizers_names():
            notifications.send_telegram_image(
                token=token,
                chat_id=chat_id,
                image_path='./images/{}'.format(opt_name),
                caption='-- {} --'.format(opt_name))

        notifications.send_telegram_image(
            token=token,
            chat_id=chat_id,
            image_path='./images/optimizers_comparison.jpg',
            caption='-- OPTIMIZERS COMPARISON --')


if __name__ == "__main__":
    main()
