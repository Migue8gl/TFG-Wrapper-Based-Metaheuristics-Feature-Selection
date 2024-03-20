from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

plt.style.use(["science", "ieee"])  # Style of plots

# ---------------------------- PLOTTING FUNCTIONS ------------------------------ #


def plot_training_curves(fitness_values: dict,
                         ax: Optional[matplotlib.axes.Axes] = None,
                         title: Optional[str] = None):
    """
    Generate a plot of fitness values over training iterations.

    Parameters:
        - fitness_values (dict): A dictionary containing fitness values for training and validation.
        - ax (matplotlib.axes.Axes, optional): The axes on which to plot the fitness values.
        - title (str, optional): The title of the plot.
    """

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(fitness_values, label="Training Fitness", color="blue")
    if title is None:
        ax.set_title("Training curves")
    else:
        ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness Value")
    ax.legend()


def plot_fitness_over_folds(fitness_values: dict,
                            iterations: int,
                            k: int,
                            ax: Optional[matplotlib.axes.Axes] = None,
                            title: str = None):
    """
    Generate a plot of fitness values over folds.

    Parameters:
        - fitness_values (dict): A dictionary containing fitness values for training and validation.
        - iterations (int): The number of iterations.
        - k (int): The number of folds.
        - ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided, a new figure and axes will be created.
        - title (str, optional): The title of the plot. If not provided, a default title will be used.
    """
    iteration_numbers = np.linspace(0, iterations + 1,
                                    len(fitness_values["avg_fitness"]))
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(
        iteration_numbers,
        fitness_values['avg_fitness'],
        label="Fitness",
        color="blue",
    )
    if title is None:
        ax.set_title("Average fitness {}-fold cross validation".format(k))
    else:
        ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness Value")
    metrics = "Average: {:.2f}\nStandard Deviation: {:.2f}".format(
        fitness_values['test_fitness']['avg'],
        fitness_values['test_fitness']['std_dev'])

    ax.text(0.7,
            0.5,
            metrics,
            transform=ax.transAxes,
            bbox=dict(facecolor='white',
                      edgecolor='black',
                      boxstyle='round,pad=0.5'))
    ax.legend(loc='upper right')


def plot_fitness_over_population_sizes(
        fitness_values: list,
        population_sizes: list,
        ax: Optional[matplotlib.axes.Axes] = None,
        title: str = None):
    """
    Plots the fitness values over different population sizes.

    Parameters:
        - fitness_values (list): A list of fitness values.
        - population_sizes (list): A list of population sizes.
        - ax (matplotlib.axes.Axes, optional): The axes on which to plot the fitness values. If not provided, a new figure and axes will be created.
        - title (str, optional): The title of the plot. If not provided, a default title will be set.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(population_sizes,
            fitness_values,
            label="Fitness",
            color="purple",
            marker="d")
    if title is None:
        ax.set_title("Fitness test value over population sizes")
    else:
        ax.set_title(title)
    ax.set_xlabel("Population Size")
    ax.set_ylabel("Fitness Value")
    ax.legend()


def plot_fitness_all_optimizers(optimizers_fitness: dict,
                                iterations: list,
                                ax: Optional[matplotlib.axes.Axes] = None,
                                title: str = None):
    """
    Plots the fitness values of multiple optimizers over a specified number of iterations.

    Parameters:
        - optimizers_fitness (dict): A dictionary containing the fitness values of each optimizer. The keys are the names of the optimizers and the values are lists of fitness values.
        - iterations (int): The total number of iterations.
        - ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot the fitness values on. If not provided, a new figure and axes will be created.
        - title (str, optional): The title of the plot. If not provided, a default title will be used.
    """
    iteration_numbers = np.linspace(
        0, iterations + 1, len(next(iter(optimizers_fitness.values()))))
    if ax is None:
        _, ax = plt.subplots()
    for key, fitness_values in optimizers_fitness.items():
        ax.plot(iteration_numbers, fitness_values, label=key, marker="X")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    if title is None:
        ax.set_title("Optimizer Comparison - Fitness Over Time")
    else:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)


def plot_s_shaped_transfer_function(ax: Optional[matplotlib.axes.Axes] = None,
                                    title: str = None):
    """
    Plots the s-shape transfer function.

    Parameters:
        - ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot the fitness values on. If not provided, a new figure and axes will be created.
        - title (str, optional): The title of the plot. If not provided, a default title will be used.
    """
    if ax is None:
        _, ax = plt.subplots()
    x = np.linspace(-5, 5, 100)
    y = 1 / (1 + np.exp(-x))
    ax.plot(x, y, color="darkgreen", label=r"$\frac{1}{1 + e^{-x}}$")
    ax.set_xlabel("x")
    ax.set_ylabel("T(x)")
    if title is None:
        ax.set_title("S-Shaped Transfer Function")
    else:
        ax.set_title(title)
    ax.legend()


def plot_v_shaped_transfer_function(ax: Optional[matplotlib.axes.Axes] = None,
                                    title: str = None):
    """
    Plots the s-shape transfer function.

    Parameters:
        - ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot the fitness values on. If not provided, a new figure and axes will be created.
        - title (str, optional): The title of the plot. If not provided, a default title will be used.
    """
    if ax is None:
        _, ax = plt.subplots()
    x = np.linspace(-5, 5, 100)
    y = np.abs(np.tanh(x))
    ax.plot(x, y, color="red", label=r"$|\tanh(x)|$")
    ax.set_xlabel("x")
    ax.set_ylabel("T(x)")
    if title is None:
        ax.set_title("V-Shaped Transfer Function")
    else:
        ax.set_title(title)
    ax.legend()
