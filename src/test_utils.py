from typing import Optional

import matplotlib.pyplot as plt
from analysis_utils import k_fold_cross_validation
from constants import (
    D2,  # noqa: F401
    D1,  # noqa: F401
    D3,  # noqa: F401
    D5,  # noqa: F401
    D6,  # noqa: F401
    D7,  # noqa: F401
    D8,  # noqa: F401
    D9,  # noqa: F401
    D10,  # noqa: F401
    D11,  # noqa: F401
    DATA,
    DEFAULT_FOLDS,
    DEFAULT_OPTIMIZER,
    DEFAULT_TEST_ITERATIONS,
    LABELS,
    SAMPLE,
)
from data_utils import load_data, scaling_min_max, split_data_to_dict, scaling_std_score  # noqa: F401
from optimizer import Optimizer
from plots import plot_metrics_over_folds, plot_training_curves
from sklearn.model_selection import train_test_split


def default_parameters(opt: Optional[str] = None,
                       dataset_path: Optional[str] = None):
    """
    Generates a dictionary with default parameters testing purposes.

    Parameters:
        - opt (str, optional): Name of the optimizer's parameters to return.

    Returns:
        - parameters (dict): A dictionary containing default parameters for testing:
            - 'k' (int): The number of fold to consider in k-fold cross valitation.
            - 'optimizer' (object): The optimizer object used for optimizing the algorithm.
            - 'dataset_dict' (dict): The dataset in dict form splitted into samples and labels.
               
    """

    # Test parameters
    # Load, normalize and split dataset into samples and labels
    dataset = split_data_to_dict(load_data(dataset_path))

    # Catching optimizer default parameters
    optimizer_parameters = Optimizer.get_default_optimizer_parameters(
        opt.lower() if opt else DEFAULT_OPTIMIZER, dataset[SAMPLE].shape[1])

    # Some optimizers have generations instead of iterations
    if "iterations" in optimizer_parameters:
        optimizer_parameters["iterations"] = DEFAULT_TEST_ITERATIONS
    elif "generations" in optimizer_parameters:
        optimizer_parameters["generations"] = DEFAULT_TEST_ITERATIONS

    optimizer = Optimizer(opt if opt else DEFAULT_OPTIMIZER,
                          optimizer_parameters)

    return {'k': DEFAULT_FOLDS, 'optimizer': optimizer, 'dataset': dataset}


def test_run_optimizer(optimizer: object, dataset: Optional[dict] = None):
    """
    Run the optimizer and plot the fitness curves over training.

    Parameters:
        - optimizer (object): The optimizer to be used.
        - dataset (dict): The dataset in dict form splitted into samples and labels.
    """
    # Split into train and test data
    x_train, x_test, y_train, y_test = train_test_split(dataset[SAMPLE],
                                                        dataset[LABELS],
                                                        test_size=0.2,
                                                        random_state=42)

    # Creating the problem for the Optimizer
    problem_to_optimize = {SAMPLE: x_train, LABELS: y_train}

    # Running the optimizer
    best_result, fitness_values = optimizer.optimize(problem_to_optimize)

    # Printing best result achived in training
    print("Best result for {} optimizer in {} classifier: {}".format(
        optimizer.name,
        optimizer.params['target_function_parameters']["classifier"],
        round(best_result[-1], 2),
    ))

    # Testing the best result
    optimizer.params['target_function_parameters'][DATA] = {
        SAMPLE: x_test,
        LABELS: y_test
    }
    optimizer.params['target_function_parameters'][
        'weights'] = best_result[:-2]
    test = Optimizer.fitness(
        **optimizer.params['target_function_parameters'])['val_fitness']

    # Printing test results
    print("Test result: {}".format(round(test, 2)))

    # Plotting average fitness curves
    plot_training_curves(fitness_values=fitness_values,
                         title="Training curve on {} optimizer".format(
                             optimizer.name))

    plt.savefig("./images/test_optimizer_training.jpg")


def test_cross_validation(optimizer: object,
                          dataset: Optional[dict] = None,
                          k: int = DEFAULT_FOLDS):
    """
    Test cross-validation function.

    Parameters:
        - optimizer (object): The optimizer to be used.
        - dataset (dict): The dataset in dict form splitted into samples and labels.
        - k (int): The number of folds for cross-validation. Default is 5.
    """
    # Test cross validation
    metrics = k_fold_cross_validation(dataset=dataset,
                                      optimizer=optimizer,
                                      k=k,
                                      scaler=2,
                                      verbose=True)

    print(
        'Average test fitness over {} Folds for {} optimizer ({}): {}'.format(
            k, optimizer.name,
            optimizer.params['target_function_parameters']['classifier'],
            round(metrics['test_fitness']['avg'], 2)))

    # Plotting average fitness over k folds in cross validation
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    plot_metrics_over_folds(
        metrics, optimizer.params['iterations'] if 'iterations'
        in optimizer.params else optimizer.params['generations'], k, ax,
        'Average fitness {}-fold cross validation'.format(k))

    fig.suptitle('TEST RUNNING {} ON {}-FOLD CROSS VALIDATION WITH {}'.format(
        optimizer.name, k,
        optimizer.params['target_function_parameters']['classifier'].lower()),
                 fontweight='bold',
                 fontsize=16)
    plt.tight_layout()
    plt.savefig('./images/test_k_cross_validation.jpg')


if __name__ == "__main__":
    optimizer = 'goa'
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plot_s_shaped_transfer_function(axs[0])
    plot_v_shaped_transfer_function(axs[1])
    plt.savefig("./images/transfer_functions.jpg")
    
    
    parameters = default_parameters(optimizer, D2)
    del parameters['k']
    test_run_optimizer(**parameters)
    """

    test_cross_validation(**default_parameters(optimizer, D11))
