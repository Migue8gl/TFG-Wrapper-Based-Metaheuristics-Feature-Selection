import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from data_utils import *

plt.style.use(['science', 'ieee'])  # Style of plots

# ---------------------------- PLOTTING FUNCTIONS ------------------------------ #


def plot_fitness_over_training(fitness_values, ax=None, title=None):
    """
    Generate a plot of fitness values over training iterations.

    Parameters:
    - fitness_values (dict): A dictionary containing fitness values for training and validation.
    - iterations (int): The total number of training iterations.
    - ax (matplotlib.axes._subplots.AxesSubplot): The axes on which to plot the fitness values.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    validation_fitness, training_fitness = split_dicts_keys(fitness_values)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(validation_fitness, label='Training Fitness', color='blue')
    ax.plot(training_fitness, label='Validation Fitness', color='orange')
    if title == None:
        ax.set_title('Training curves')
    else:
        ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness Value')
    ax.legend()


def plot_fitness_over_folds(fitness_values,
                            iterations,
                            k,
                            ax=None,
                            title=None):
    """
    Generate a plot of fitness values over folds.

    Parameters:
    - fitness_values (dict): A dictionary containing fitness values for training and validation.
    - iterations (int): The number of iterations.
    - k (int): The number of folds.
    - ax (AxesSubplot, optional): The axes to plot on. If not provided, a new figure and axes will be created.
    - title (str, optional): The title of the plot. If not provided, a default title will be used.

    Returns:
    - None
    """
    iteration_numbers = np.linspace(0, iterations + 1,
                                    len(fitness_values['ValFitness']))
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(iteration_numbers,
            fitness_values['TrainFitness'],
            label='Fitness',
            color='blue')
    ax.plot(iteration_numbers,
            fitness_values['ValFitness'],
            label='Validation Fitness',
            color='orange')
    if title == None:
        ax.set_title('Average fitness {}-fold cross validation'.format(k))
    else:
        ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness Value')
    ax.legend()


def plot_fitness_over_population_sizes(fitness_values,
                                       population_sizes,
                                       ax=None,
                                       title=None):
    """
    Plots the fitness values over different population sizes.

    Parameters:
    - fitness_values (list): A list of fitness values.
    - population_sizes (list): A list of population sizes.
    - ax (Axes, optional): The axes on which to plot the fitness values. If not provided, a new figure and axes will be created.
    - title (str, optional): The title of the plot. If not provided, a default title will be set.

    Returns:
    - None
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(population_sizes,
            fitness_values,
            label='Fitness',
            color='purple',
            marker='d')
    if title == None:
        ax.set_title('Fitness test value over population sizes')
    else:
        ax.set_title(title)
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Fitness Value')
    ax.legend()


def plot_fitness_all_optimizers(optimizers_fitness,
                                iterations,
                                ax=None,
                                title=None):
    """
    Plots the fitness values of multiple optimizers over a specified number of iterations.

    Parameters:
    - optimizers_fitness (dict): A dictionary containing the fitness values of each optimizer. The keys are the names of the optimizers and the values are lists of fitness values.
    - iterations (int): The total number of iterations.
    - ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot the fitness values on. If not provided, a new figure and axes will be created.
    - title (str, optional): The title of the plot. If not provided, a default title will be used.

    Returns:
    - None
    """
    iteration_numbers = np.linspace(
        0, iterations + 1, len(next(iter(optimizers_fitness.values()))))
    if ax is None:
        _, ax = plt.subplots()
    for key, fitness_values in optimizers_fitness.items():
        ax.plot(iteration_numbers, fitness_values, label=key, marker='X')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness')
    if title is None:
        ax.set_title('Optimizer Comparison - Fitness Over Time')
    else:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)
