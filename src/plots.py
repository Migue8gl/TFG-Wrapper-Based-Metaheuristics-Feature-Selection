import matplotlib.pyplot as plt
import numpy as np

def plot_fitness_over_folds(fitness_values, iterations, k, ax=None, title=None):
    iteration_numbers = np.linspace(0, iterations+1, len(fitness_values['ValFitness']))
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(iteration_numbers, fitness_values['TrainFitness'], label='Fitness', color='blue')
    ax.plot(iteration_numbers, fitness_values['ValFitness'], label='Validation Fitness', color='orange')
    if title == None:
        ax.set_title('Average fitness {}-fold cross validation'.format(k))
    else:
        ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness Value')
    ax.legend()

def plot_fitness_over_population_sizes(fitness_values, population_sizes, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(population_sizes, fitness_values, label='Fitness', color='purple', marker='d')
    if title == None:
        ax.set_title('Fitness test value over population sizes')
    else:
        ax.set_title(title)
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Fitness Value')
    ax.legend()

def plot_fitness_all_optimizers(optimizers_fitness, iterations, ax=None, title=None):
    iteration_numbers = np.linspace(0, iterations+1, len(next(iter(optimizers_fitness.values()))))
    if ax is None:
        fig, ax = plt.subplots()
    for key, fitness_values in optimizers_fitness.items():
        ax.plot(iteration_numbers, fitness_values, label=key, marker='X')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness')
    if title is None:
        ax.set_title('Optimizator Comparison - Fitness Over Time')
    else:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)