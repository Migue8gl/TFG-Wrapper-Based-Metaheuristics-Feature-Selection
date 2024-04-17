from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval

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


def plot_metric_over_folds(metric_values: dict,
                           metric_name: str,
                           iterations: int,
                           k: int,
                           color: str,
                           ax: Optional[plt.Axes] = None,
                           title: Optional[str] = None):
    """
    Generate a plot of a specific metric values over folds.

    Parameters:
        - metric_values (dict): A dictionary containing metric values for k-fold cross validation.
        - metric_name (str): The name of the metric to plot.
        - iterations (int): The number of iterations.
        - k (int): The number of folds.
        - color (str): The color of the plotted line.
        - ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided, a new figure and axes will be created.
        - title (str, optional): The title of the plot. If not provided, a default title will be used.
    """
    iteration_numbers = np.linspace(0, iterations + 1,
                                    len(metric_values[metric_name]))
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(
        iteration_numbers,
        metric_values[metric_name],
        label=metric_name,
        color=color,
    )
    if title is None:
        ax.set_title("{} over {}-fold cross validation".format(metric_name, k))
    else:
        ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel(metric_name)
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


def plot_grouped_boxplots(data,
                          x='dataset',
                          y='all_fitness',
                          filter=None,
                          title='Boxplot',
                          xlabel='Dataset',
                          ylabel='Average'):
    """
    Plot grouped boxplots for the given data.

    Parameters:
        - data (pd.DataFrame): DataFrame containing the data to be plotted.
        - x (str): Name of the column grouping the data.
        - y (str): Name of the column containing values.
        - filter (dict): Name of the column to filter the data on.
        - title (str): Title of the plot.
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the y-axis.
    """
    # Filter data:
    if filter is not None:
        filtered_data = data.copy()  # Make a copy of the original data
        for col, val in filter.items():
            filtered_data = filtered_data[filtered_data[col] == val]
    else:
        filtered_data = data

    # Group the data by dataset
    grouped_data = filtered_data.groupby(x)

    # Sort the groups by average value
    grouped_data = sorted(
        grouped_data, key=lambda x: np.mean(literal_eval(x[1][y].values[0])))

    # Prepare data for plotting
    data_to_plot = [
        literal_eval(data[y].values[0]) for _, data in grouped_data
    ]

    # Create a single figure
    fig = plt.figure(figsize=(12, 8))

    # Plot all boxplots together
    bp = plt.boxplot(data_to_plot,
                     patch_artist=True,
                     showmeans=True,
                     meanprops=dict(marker='^',
                                    markeredgecolor='green',
                                    markerfacecolor='green'))

    # Plot mean as a line
    m1 = [np.mean(literal_eval(data[y].values[0])) for _, data in grouped_data]
    st1 = [np.std(literal_eval(data[y].values[0])) for _, data in grouped_data]
    # Set title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set x-axis labels
    plt.xticks(range(1,
                     len(grouped_data) + 1),
               [dataset for dataset, _ in grouped_data],
               rotation=45)

    # Annotate each median line
    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        plt.annotate(f'$\\mu={m1[i]:.2f}$' + f'\n$\\sigma={st1[i]:.2f}$',
                     xy=(x, y),
                     xytext=(16, 5),
                     textcoords='offset points',
                     ha='center',
                     va='bottom',
                     fontsize=6.5)

    # Show grid
    plt.grid(True)
    plt.tight_layout()
    return fig
