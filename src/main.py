import os
import re
import sys
import time

import matplotlib.pyplot as plt
import notifications
import pandas as pd
from analysis_utils import (
    evaluate_optimizer,
)
from constants import (
    CREDENTIALS_DIR,
    D2,
    DEFAULT_EVALS,
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

plt.style.use(['science', 'ieee'])  # Style of plots


def main(*args, **kwargs):
    start_time = time.time()

    # Get parameters from the user
    dataset_arg = kwargs.get('-d', D2)  # Chosen dataset
    # Notifications
    notify_arg = kwargs.get('-n', False)
    i_arg = kwargs.get(
        '-i',
        DEFAULT_EVALS)  # Number of evaluations for each optimization process
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

    i = i_arg  # Evaluations

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

    encoding = 'real' if ('binary' in optimizer.params and
                          (optimizer.params['binary'] == 'r'
                           or not optimizer.params['binary'])) else 'binary'

    name_pattern = r'/([^/]+)\.arff$'
    dataset_name = re.search(name_pattern, dataset_arg).group(1)

    # SVC Cross validation
    metrics_svc = evaluate_optimizer(dataset=dataset_dict,
                                     optimizer=optimizer,
                                     n=i,
                                     scaler=scaling_arg,
                                     verbose=verbose_arg)

    file_path = os.path.join(RESULTS_DIR, encoding, dataset_name, "all_fitness_svc.csv")

    with open(file_path, 'a' if os.path.exists(file_path) else 'w') as file:
        file.write(f"{optimizer_arg}: {metrics_svc['avg_fitness']}\n")

    optimizer.params['target_function_parameters']['classifier'] = 'knn'
    metrics_knn = evaluate_optimizer(dataset=dataset_dict,
                                     optimizer=optimizer,
                                     n=i,
                                     scaler=scaling_arg,
                                     verbose=verbose_arg)

    file_path = os.path.join(RESULTS_DIR, encoding, f"{dataset_name}_knn.csv")

    with open(file_path, 'a' if os.path.exists(file_path) else 'w') as file:
        file.write(f"{optimizer_arg}: {metrics_knn['avg_fitness']}\n")

    # Create directory to store dataset metrics images
    img_directory_path = os.path.join(IMG_DIR, encoding, dataset_name)
    if not os.path.isdir(img_directory_path):
        os.makedirs(img_directory_path)

    # Create directory to store dataset metrics retults
    result_path = os.path.join(RESULTS_DIR, encoding, dataset_name)

    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    data = {
        'classifier': ['knn', 'svc'],
        'all_fitness': [
            metrics_knn['test_fitness']['all_fitness'],
            metrics_svc['test_fitness']['all_fitness']
        ],
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
        'all_fitness',
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

    # First set of plots
    second_key = list(parameters.keys())[1]

    plot_metric_over_folds(
        metrics_svc,
        'avg_fitness',
        parameters[second_key],
        i,
        'blue',
        ax=None,
        title='Average fitness over {} evaluations running {} (SVC)'.format(
            i, optimizer_arg))

    plt.tight_layout()
    plt.savefig(
        os.path.join(IMG_DIR, encoding, dataset_name) +
        '/{}_fitness_over_{}_evaluations_{}_{}_{}.jpg'.format(
            'SVC', i, optimizer_arg, encoding, dataset_name))

    plot_metric_over_folds(
        metrics_knn,
        'avg_fitness',
        parameters[second_key],
        i,
        'blue',
        ax=None,
        title='Average fitness over {} evaluations running {} (KNN)'.format(
            i, optimizer_arg))

    plt.tight_layout()
    plt.savefig(
        os.path.join(IMG_DIR, encoding, dataset_name) +
        '/{}_fitness_over_{}_evaluations_{}_{}_{}.jpg'.format(
            'KNN', i, optimizer_arg, encoding, dataset_name))

    plot_metric_over_folds(
        metrics_svc,
        'avg_selected_features',
        parameters[second_key],
        i,
        'orange',
        ax=None,
        title='Average selected features over {} evaluations running {} (SVC)'.
        format(i, optimizer_arg))

    plt.tight_layout()
    plt.savefig(
        os.path.join(IMG_DIR, encoding, dataset_name) +
        '/{}_n_features_over_{}_evaluations_{}_{}_{}.jpg'.format(
            'SVC', i, optimizer_arg, encoding, dataset_name))

    plot_metric_over_folds(
        metrics_svc,
        'avg_selected_features',
        parameters[second_key],
        i,
        'orange',
        ax=None,
        title='Average selected features over {} evaluations running {} (KNN)'.
        format(i, optimizer_arg))

    plt.tight_layout()
    plt.savefig(
        os.path.join(IMG_DIR, encoding, dataset_name) +
        '/{}_n_features_{}_evaluations_{}_{}_{}.jpg'.format(
            'KNN', i, optimizer_arg, encoding, dataset_name))

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
            image_path=os.path.join(IMG_DIR, encoding, dataset_name) +
            '/{}_fitness_over_{}_evaluations_{}_{}_{}.jpg'.format(
                'SVC', i, optimizer_arg, encoding, dataset_name),
            caption='-- SVC fitness_over_{}_evaluations_{}_{}_{} --'.format(
                i, optimizer_arg, encoding, dataset_name))

        notifications.send_telegram_image(
            token=token,
            chat_id=chat_id,
            image_path=os.path.join(IMG_DIR, encoding, dataset_name) +
            '/{}_n_features_over_{}_evaluations_{}_{}_{}.jpg'.format(
                'SVC', i, optimizer_arg, encoding, dataset_name),
            caption='-- SVC n_features_over_{}_evaluations_{}_{}_{} --'.format(
                i, optimizer_arg, encoding, dataset_name))

        notifications.send_telegram_image(
            token=token,
            chat_id=chat_id,
            image_path=os.path.join(IMG_DIR, encoding, dataset_name) +
            '/{}_fitness_over_{}_evaluations_{}_{}_{}.jpg'.format(
                'KNN', i, optimizer_arg, encoding, dataset_name),
            caption='-- KNN fitness_over_{}_evaluations_{}_{}_{} --'.format(
                i, optimizer_arg, encoding, dataset_name))

        notifications.send_telegram_image(
            token=token,
            chat_id=chat_id,
            image_path=os.path.join(IMG_DIR, encoding, dataset_name) +
            '/{}_n_features_over_{}_evaluations_{}_{}_{}.jpg'.format(
                'KNN', i, optimizer_arg, encoding, dataset_name),
            caption='-- KNN n_features_over_{}_evaluations_{}_{}_{} --'.format(
                i, optimizer_arg, encoding, dataset_name))

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
