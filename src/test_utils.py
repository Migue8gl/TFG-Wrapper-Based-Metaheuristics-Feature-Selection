from constants import *
import numpy as np
from data_utils import *
from analysis_utils import *
from plots import *
import matplotlib.pyplot as plt
from math import sqrt
from typing import Optional
from optimizer import Optimizer


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
    dataset = split_data_to_dict(
        scaling_min_max(scaling_std_score(load_arff_data(dataset_path))))

    # Catching optimizer default parameters
    optimizer_parameters = Optimizer.get_default_optimizer_parameters(
        opt.upper() if opt else DEFAULT_OPTIMIZER, dataset[SAMPLE].shape[1])

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

    # https://stackoverflow.com/questions/11568897/value-of-k-in-k-nearest-neighbor-algorithm
    # Empiric k used in Knn classifier, it must be overridden due to current dataset
    optimizer.params['target_function_parameters']['n_neighbors'] = int(
        sqrt(x_train.shape[0]))

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
    test = fitness(
        **optimizer.params['target_function_parameters'])['ValFitness']

    # Printing test results
    print("Test result: {}".format(round(test, 2), ))

    # Plotting average fitness curves
    plot_training_curves(fitness_values=fitness_values,
                         title="Training curve on {} optimizer".format(
                             optimizer.name))

    plt.savefig("./images/test_optimizer_training.jpg")


def test_cross_validation(optimizer: object,
                          dataset: Optional[dict] = None,
                          k: int = 5):
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
                                      k=k)

    print(
        'Average test fitness over {} Folds for {} optimizer ({}): {}'.format(
            k, optimizer.name,
            optimizer.params['target_function_parameters']['classifier'],
            round(metrics['TestFitness']['Average'])))

    # Plotting average fitness over k folds in cross validation
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    plot_fitness_over_folds(
        metrics, optimizer.params['iterations'], k, ax,
        'Average fitness {}-fold cross validation'.format(k))

    fig.suptitle('TEST RUNNING {} ON {}-FOLD CROSS VALIDATION WITH {}'.format(
        optimizer.name, k,
        optimizer.params['target_function_parameters']['classifier'].upper()),
                 fontweight='bold',
                 fontsize=16)
    plt.tight_layout()
    plt.savefig('./images/test_k_cross_validation.jpg')


if __name__ == "__main__":
    optimizer = "GWO"
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plot_s_shaped_transfer_function(axs[0])
    plot_v_shaped_transfer_function(axs[1])
    plt.savefig("./images/transfer_functions.jpg")
    
    test_run_optimizer(
        **{
            key: value
            for key, value in default_parameters(optimizer, D2).items()
            if key != "k"
        })
    """

    test_cross_validation(**default_parameters(optimizer, D2))
