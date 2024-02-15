from constants import *
import numpy as np
from data_utils import *
from analysis_utils import *
from plots import *
import matplotlib.pyplot as plt
from math import sqrt
from typing import Optional
from optimizer import Optimizer


def default_parameters(opt: Optional[str] = None):
    """
    Generates a dictionary with default parameters testing purposes.

    Parameters:
        - opt (str, optional): Name of the optimizer's parameters to return.

    Returns:
        - parameters (dict): A dictionary containing default parameters for testing:
            - 'k' (int): The number of neighbors to consider in the algorithm.
            - 'optimizer' (object): The optimizer object used for optimizing the algorithm.
               
    """

    # Test parameters
    dataset = split_data_to_dict(scaling_min_max(load_arff_data(D2)))
    optimizer_parameters = Optimizer.get_default_optimizer_parameters(
        opt.upper() if opt else DEFAULT_OPTIMIZER, dataset[SAMPLE].shape[1])[0]
    if "iterations" in optimizer_parameters:
        optimizer_parameters["iterations"] = DEFAULT_TEST_ITERATIONS
    elif "generations" in optimizer_parameters:
        optimizer_parameters["generations"] = DEFAULT_TEST_ITERATIONS

    optimizer_parameters['target_function_parameters']['data'] = dataset
    optimizer = Optimizer(opt if opt else DEFAULT_OPTIMIZER,
                          optimizer_parameters)

    return {"k": DEFAULT_FOLDS, "optimizer": optimizer}


def test_run_optimizer(opt: Optional[str] = None,
                       dataset: Optional[dict] = None):
    """
    Run the optimizer and plot the fitness curves over training.

    Parameters:
        - optimizer (str, optional): The optimizer to be used.
    """
    optimizer_parameters = Optimizer.get_default_optimizer_parameters(opt, dataset[SAMPLE].shape[1])
    dataset = optimizer_parameters['target_function_parameters']['data']

    x_train, x_test, y_train, y_test = train_test_split(dataset[SAMPLE],
                                                        dataset[LABELS],
                                                        test_size=0.2,
                                                        random_state=42)
    
    optimizer_parameters['target_function_parameters']['data'] = {SAMPLE: x_train, LABELS: y_train}
    
    optimizer = Optimizer(opt if opt else DEFAULT_OPTIMIZER,
                          optimizer_parameters)
    target_function_parameters[SAMPLE] = {SAMPLE: x_train, LABELS: y_train}

    # https://stackoverflow.com/questions/11568897/value-of-k-in-k-nearest-neighbor-algorithm
    target_function_parameters['n_neighbors'] = int(sqrt(x_train.shape[0]))

    # Running the optimizer
    optimizer.params
    best_result, fitness_values = optimizer.optimize()

    print("Best result for {} optimizer in {} classifier: {}".format(
        optimizer.name,
        target_function_parameters["classifier"],
        round(best_result[-1], 2),
    ))

    target_function_parameters[SAMPLE] = {SAMPLE: x_test, LABELS: y_test}
    target_function_parameters['weights'] = best_result[:-2]
    test = fitness(**target_function_parameters)['ValFitness']

    print("Test result: {}".format(round(test, 2), ))

    # Plotting average fitness curves
    plot_training_curves(
        fitness_values=fitness_values,
        title="Training curve on {} optimizer".format(optimizer_name))

    plt.savefig("./images/test_optimizer_training.jpg")


def test_cross_validation(
    k=5,
    dataset=None,
    optimizer=OPTIMIZERS[DEFAULT_OPTIMIZER],
    optimizer_parameters=None,
    target_function_parameters=None,
):
    """
    Test cross-validation function.

    Parameters:
    - k (int): The number of folds for cross-validation. Default is 5.
    - dataset (dict): The dataset to use for cross-validation. Default is None.
    - optimizer (object): The optimizer to use for optimization. Default is OPTIMIZERS[DEFAULT_OPTIMIZER].
    - optimizer_parameters (dict): The parameters for the optimizer. Default is None.
    - target_function_parameters (dict): The parameters for the target function. Default is None.

    Returns:
    - None
    """
    optimizer_name = get_optimizer_name_by_function(optimizer)

    # Test cross validation
    test_fitness, metrics = k_fold_cross_validation(
        dataset=dataset,
        optimizer=optimizer,
        k=k,
        parameters=optimizer_parameters,
        target_function_parameters=target_function_parameters,
    )
    print(
        "Average test fitness over {} Folds for {} optimizer ({}): {}".format(
            k,
            optimizer_name,
            target_function_parameters["classifier"],
            round(np.mean(test_fitness), 2),
        ))

    # Plotting average fitness over k folds in cross validation
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    plot_fitness_over_folds(
        metrics,
        optimizer_parameters["iterations"],
        k,
        ax,
        "Average fitness {}-fold cross validation".format(k),
    )

    fig.suptitle(
        "TEST RUNNING {} ON {}-FOLD CROSS VALIDATION".format(
            optimizer_name, k),
        fontweight="bold",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(
        "./images/test_k_cross_validation_{}.jpg".format(optimizer_name))


if __name__ == "__main__":
    optimizer = "ACO"
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plot_s_shaped_transfer_function(axs[0])
    plot_v_shaped_transfer_function(axs[1])
    plt.savefig("./images/transfer_functions.jpg")
    
    """
    test_run_optimizer(
        **{
            key: value
            for key, value in default_parameters(optimizer).items()
            if key != "k"
        })
    """

    test_cross_validation(**default_parameters(optimizer))

    """
