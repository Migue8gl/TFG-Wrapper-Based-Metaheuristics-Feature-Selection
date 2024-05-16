from ast import literal_eval
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from constants import (
    KNN_CLASSIFIER,
    OPTIMIZER_COLOR,
    RESULTS_DIR,
    SVC_CLASSIFIER,
)

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
        title: Optional[str] = None):
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
                                ax: Optional[matplotlib.axes.Axes] = None,
                                title: Optional[str] = None,
                                optimizer_color: dict = OPTIMIZER_COLOR):
    """
    Plots the fitness values of multiple optimizers over a specified number of iterations.

    Parameters:
        - optimizers_fitness (dict): A dictionary containing the fitness values of each optimizer. The keys are the names of the optimizers and the values are lists of fitness values.
        - ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot the fitness values on. If not provided, a new figure and axes will be created.
        - title (str, optional): The title of the plot. If not provided, a default title will be used.
        - optimizer_color (dict, optional): A dictionary containing the colors of each optimizer. The keys are the names of the optimizers and the values are the corresponding colors.
    """

    if ax is None:
        _, ax = plt.subplots()
    for key, fitness_values in optimizers_fitness.items():
        iteration_numbers = np.arange(0, len(optimizers_fitness[key.lower()]))
        ax.plot(iteration_numbers,
                fitness_values,
                label=key,
                c=optimizer_color[key.lower()])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    if title is None:
        ax.set_title("Optimizer Comparison - Fitness Over Time")
    else:
        ax.set_title(title)
    ax.legend(loc='upper right', fontsize=4, ncol=2)
    ax.grid(True)


def plot_s_shaped_transfer_function(ax: Optional[matplotlib.axes.Axes] = None,
                                    title: Optional[str] = None):
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
                                    title: Optional[str] = None):
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


def plot_grouped_boxplots(data: pd.DataFrame,
                          x: str = 'dataset',
                          x_color: dict = OPTIMIZER_COLOR,
                          y: str = 'all_fitness',
                          filter: Optional[dict] = None,
                          title: str = 'Boxplot',
                          xlabel: str = 'Dataset',
                          ylabel: str = 'Average') -> matplotlib.figure.Figure:
    """
    Plot grouped boxplots for the given data.

    Parameters:
        - data (pd.DataFrame): DataFrame containing the data to be plotted.
        - x (str): Name of the column grouping the data.
        - x_color (dict): Color associated with each value in the x column.
        - y (str): Name of the column containing values.
        - filter (dict): Name of the column to filter the data on.
        - title (str): Title of the plot.
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the y-axis.

    Returns:
        - fig (matplotlib.figure.Figure): The created figure.
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

    # Get the keys of the x column
    x_keys = [group[0] for group in grouped_data]

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
                                    color='gold',
                                    markeredgecolor='darkgoldenrod',
                                    markerfacecolor='gold',
                                    markersize=10),
                     medianprops=dict(color='red', markersize=15))

    # Assign colors to boxplots
    for col, box in zip(x_keys, bp['boxes']):
        box.set(color=x_color[col])

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


def plot_rankings(ranking: pd.DataFrame,
                  title: str = 'Ranking',
                  optimizer_color: dict = OPTIMIZER_COLOR):
    """
    Plots rankings as a bar chart with a different color for each optimizer.

    Parameters:
        - ranking (pandas.DataFrame): Rankings as a DataFrame.
        - title (str): Title of the plot.
        - optimizer_color (dict): Dictionary with optimizer:color pairs.
    """
    plt.figure(figsize=(10, 8))  # Adjust the figure size if needed

    x_values = ranking.columns.tolist()[1:]  # Get optimizer names
    y_values = ranking.iloc[-1, 1:].values  # Get mean rankings

    for x, y in sorted(zip(x_values, y_values), key=lambda x: x[1]):
        color = optimizer_color[x]
        plt.bar(x, y, color=color,
                label=f'{x}: {y:.2f}')  # Show ranking with two decimals

    plt.title(title)
    plt.xlabel('Optimizer')
    plt.ylabel('Mean Ranking')
    plt.legend()  # Show legend with optimizer names and their rankings
    plt.xticks(rotation=45,
               ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()


def plot_all_boxplots_optimizers(df_analysis_b: pd.DataFrame,
                                 df_analysis_r: pd.DataFrame,
                                 optimizer_color: dict = OPTIMIZER_COLOR):
    """
    Plots boxplots for all optimizers for a given dataset and classifier.

    Parameters:
        - df_analysis_b (pandas.DataFrame): Dataframe with binary analysis results.
        - df_analysis_r (pandas.DataFrame): Dataframe with real analysis results.
        - classifiers (list): List of classifiers to plot.
        - optimizer_color (dict): Dictionary with optimizer:color pairs.
    """

    def _plot_optimizers(df_encoding: pd.DataFrame, encoding: str,
                         dataset_name: str, classifier: str):
        filter_dict = {'dataset': dataset_name, 'classifier': classifier}

        fig_fitness = plot_grouped_boxplots(
            df_encoding,
            x='optimizer',
            x_color=optimizer_color,
            filter=filter_dict,
            title=
            f'Boxplot Grouped by Optimizer - {encoding} - {classifier} - {dataset_name}',
            ylabel='Average Fitness')
        plt.savefig(
            f'{RESULTS_DIR}{encoding}/{dataset_name}/optimizer_boxplot_fitness_{classifier}_{encoding[0]}.png'
        )
        plt.close(fig_fitness)

    classifiers = [SVC_CLASSIFIER, KNN_CLASSIFIER]
    for encoding in ['binary', 'real']:
        df_encoding = df_analysis_b if encoding == 'binary' else df_analysis_r
        for dataset_name in df_encoding['dataset'].unique():
            for classifier in classifiers:
                _plot_optimizers(df_encoding, encoding, dataset_name,
                                 classifier)
