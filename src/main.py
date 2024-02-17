import numpy as np
from constants import *
import matplotlib.pyplot as plt
from data_utils import *
import time
from analysis_utils import k_fold_cross_validation, anaysis_fitness_over_population, get_optimizer_parameters, analysis_optimizers_comparison
from plots import plot_fitness_over_folds, plot_fitness_over_population_sizes, plot_fitness_all_optimizers
from optimizer import Optimizer
import notifications

plt.style.use(['science', 'ieee'])  # Style of plots


def main(*args, **kwargs):
    start_time = time.time()

    # Get parameters from the user
    dataset_arg = kwargs.get('-d', D2)  # Chosen dataset
    # Notifications meaning end of training
    notify_arg = kwargs.get('-n', False)
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
        'C': 0.1
    }

    for opt in Optimizer.get_optimizers_names():
        # Optimization function's parameters
        parameters = Optimizer.get_default_optimizer_parameters(
            opt.upper(), dataset_dict[SAMPLE].shape[1])
        # Creating optimizer object
        optimizer = Optimizer(opt, parameters)

        # SVC Cross validation for x optimizer alone
        metrics = k_fold_cross_validation(dataset=dataset_dict,
                                          optimizer=optimizer,
                                          k=k)

        # Test for optimizer X altering population size with SVC
        fitness_over_different_populations = anaysis_fitness_over_population(
            dataset=dataset_dict, optimizer=optimizer, k=k)

        # KNN cross validation for optimizer X alone
        optimizer.params['target_function_parameters']['classifier'] = 'knn'
        metrics2 = k_fold_cross_validation(dataset=dataset_dict,
                                           optimizer=optimizer,
                                           k=k)

        # Test for optimizer X altering population size with KNN
        fitness_over_different_populations2 = anaysis_fitness_over_population(
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
        token, chat_id = notifications.load_credentials('./credentials/credentials.txt')
        
        notifications.send_telegram_message(
            token=token,
            chat_id=chat_id,
            message='### Ejecución Terminada - Tiempo total {} segundos ###'.
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
