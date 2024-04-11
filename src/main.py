import os
import re
import sys
import time

import matplotlib.pyplot as plt
import notifications
import pandas as pd
from analysis_utils import (
    k_fold_cross_validation,
)
from constants import (
    CREDENTIALS_DIR,
    D2,
    DEFAULT_FOLDS,
    DEFAULT_OPTIMIZER,
    IMG_DIR,
    RESULTS_DIR,
    SAMPLE,
)
from data_utils import (
    load_data,
    split_data_to_dict,
)
from optimizer import Optimizer
from plots import (
    plot_metric_over_folds,
)

#plt.style.use(['science', 'ieee'])  # Style of plots


def main(*args, **kwargs):
    start_time = time.time()

    # Get parameters from the user
    dataset_arg = kwargs.get('-d', D2)  # Chosen dataset
    # Notifications
    notify_arg = kwargs.get('-n', False)
    k_arg = kwargs.get('-k',
                       DEFAULT_FOLDS)  # Number of folds in cross validation
    scaling_arg = kwargs.get('-s', 1)  # Type of scaling applied to dataset
    optimizer_arg = kwargs.get('-o', DEFAULT_OPTIMIZER).lower()
    verbose_arg = kwargs.get('-v', False)
    binary_arg = kwargs.get(
        '-b',
        's')  # Use optimizer with binary encoding using transfer functions

    # Core functionality
    dataset = load_data(dataset_arg)

    # Split the data into dict form
    dataset_dict = split_data_to_dict(dataset)

    k = k_arg  # F fold cross validation

    # Optimization function's parameters
    parameters = Optimizer.get_default_optimizer_parameters(
        optimizer_arg.lower(), dataset_dict[SAMPLE].shape[1])

    # Creating optimizer object
    optimizer = Optimizer(optimizer_arg, parameters)
    optimizer.params['verbose'] = verbose_arg

    if 'binary' in optimizer.params.keys():
        if optimizer.name == 'ga' and binary_arg != 'r':
            optimizer.params['binary'] = True
        elif optimizer.name == 'ga':
            optimizer.params['binary'] = False
        else:
            optimizer.params['binary'] = binary_arg

    encoding = 'real' if ('binary' in optimizer.params
                            and optimizer.params['binary'] == 'r') else 'binary'
    # SVC Cross validation
    metrics_svc = k_fold_cross_validation(dataset=dataset_dict,
                                          optimizer=optimizer,
                                          k=k,
                                          scaler=scaling_arg,
                                          verbose=verbose_arg)

    optimizer.params['target_function_parameters']['classifier'] = 'knn'
    metrics_knn = k_fold_cross_validation(dataset=dataset_dict,
                                          optimizer=optimizer,
                                          k=k,
                                          scaler=scaling_arg,
                                          verbose=verbose_arg)

    name_pattern = r'/([^/]+)\.arff$'
    dataset_name = re.search(name_pattern, dataset_arg).group(1)
    # Create directory to store dataset metrics images
    img_directory_path = os.path.join(IMG_DIR, dataset_name)
    if not os.path.isdir(img_directory_path):
        os.makedirs(img_directory_path)

    # Create directory to store dataset metrics retults
    result_path = os.path.join(RESULTS_DIR, encoding, dataset_name)

    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    data = {
        'classifier': ['knn', 'svc'],
        'best': [
            metrics_knn['test_fitness']['best'],
            metrics_svc['test_fitness']['best']
        ],
        'avg': [
            metrics_knn['test_fitness']['avg'],
            metrics_svc['test_fitness']['avg']
        ],
        'std_dev': [
            metrics_knn['test_fitness']['std_dev'],
            metrics_svc['test_fitness']['std_dev']
        ],
        'acc': [
            metrics_knn['test_fitness']['acc'],
            metrics_svc['test_fitness']['acc']
        ],
        'n_features': [
            metrics_knn['test_fitness']['n_features'],
            metrics_svc['test_fitness']['n_features']
        ],
        'selected_rate': [
            metrics_knn['test_fitness']['selected_rate'],
            metrics_svc['test_fitness']['selected_rate']
        ],
        'execution_time':
        [metrics_knn['execution_time'], metrics_svc['execution_time']]
    }
    columns = [
        'classifier',
        'best',
        'avg',
        'std_dev',
        'acc',
        'n_features',
        'selected_rate',
        'execution_time',
    ]

    df = pd.DataFrame(data, columns=columns)
    df = df.set_index(['classifier'])

    # Save the DataFrame to a CSV file
    df.to_csv(result_path + '/{}_{}.csv'.format(dataset_name, optimizer_arg),
              index=True)

    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    # First set of plots
    second_key = list(parameters.keys())[1]

    plot_metric_over_folds(
        metrics_svc,
        'avg_fitness',
        parameters[second_key],
        k,
        'blue',
        ax=axs[0],
        title='Average fitness {}-fold cross validation running {} (SVC)'.
        format(k, optimizer_arg))

    plot_metric_over_folds(
        metrics_knn,
        'avg_fitness',
        parameters[second_key],
        k,
        'blue',
        ax=axs[1],
        title='Average fitness {}-fold cross validation running {} (KNN)'.
        format(k, optimizer_arg))

    plt.tight_layout()
    plt.savefig(IMG_DIR + dataset_name +
                '/fitness_{}_fold_cross_validation_{}_{}_{}.jpg'.format(
                    k, optimizer_arg, encoding, dataset_name))

    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    plot_metric_over_folds(
        metrics_svc,
        'avg_selected_features',
        parameters[second_key],
        k,
        'orange',
        ax=axs[0],
        title=
        'Average selected features {}-fold cross validation running {} (SVC)'.
        format(k, optimizer_arg))

    plot_metric_over_folds(
        metrics_svc,
        'avg_selected_features',
        parameters[second_key],
        k,
        'orange',
        ax=axs[1],
        title=
        'Average selected features {}-fold cross validation running {} (KNN)'.
        format(k, optimizer_arg))

    plt.tight_layout()
    plt.savefig(IMG_DIR + dataset_name +
                '/n_features_{}_fold_cross_validation_{}_{}_{}.jpg'.format(
                    k, optimizer_arg, encoding, dataset_name))

    total_time = time.time() - start_time

    if (notify_arg):
        token, chat_id = notifications.load_credentials(CREDENTIALS_DIR +
                                                        'credentials.txt')

        notifications.send_telegram_message(
            token=token,
            chat_id=chat_id,
            message='### Execution finished - Total time {} seconds ###'.
            format(round(total_time, 4)))

        notifications.send_telegram_image(
            token=token,
            chat_id=chat_id,
            image_path=IMG_DIR + dataset_name +
            '/fitness_{}_fold_cross_validation_{}_{}_{}.jpg'.format(
                k, optimizer_arg, encoding, dataset_name),
            caption='-- fitness_{}_fold_cross_validation_{}_{}_{} --'.format(
                k, optimizer_arg, encoding, dataset_name))

        notifications.send_telegram_image(
            token=token,
            chat_id=chat_id,
            image_path=IMG_DIR + dataset_name +
            '/n_features_{}_fold_cross_validation_{}_{}_{}.jpg'.format(
                k, optimizer_arg, encoding, dataset_name),
            caption='-- n_features_{}_fold_cross_validation_{}_{}_{} --'.
            format(k, optimizer_arg, encoding, dataset_name))

        notifications.send_telegram_file(
            token=token,
            chat_id=chat_id,
            file_path=RESULTS_DIR + '/' + encoding + '/' + dataset_name +
            '/{}_{}.csv'.format(dataset_name, optimizer_arg),
            caption='-- {} {} in {} results -- '.format(
                encoding, optimizer_arg, dataset_name),
            verbose=False)


if __name__ == "__main__":
    args = sys.argv[1:]  # Skip the script name
    kwargs = {}
    for i in range(len(args)):
        if args[i].startswith('-'):
            if i + 1 < len(args) and not args[i + 1].startswith('-'):
                kwargs[args[i]] = args[i + 1]
            else:
                kwargs[args[i]] = None
    main(**kwargs)
