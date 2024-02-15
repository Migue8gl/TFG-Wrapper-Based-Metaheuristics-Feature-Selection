from pyMetaheuristic.algorithm import (
    grasshopper_optimization_algorithm, dragonfly_algorithm,
    grey_wolf_optimizer, whale_optimization_algorithm,
    artificial_bee_colony_optimization, bat_algorithm, firefly_algorithm,
    particle_swarm_optimization, genetic_algorithm, ant_colony_optimization)
from constants import *
import numpy as np


class Optimizer:
    """
    Optimizer class using the Strategy design pattern.
    """

    # ----------------------------- ATTRIBUTES ------------------------------- #

    # Optimizers dict with all available optimizers
    optimizers = {
        'GOA': grasshopper_optimization_algorithm,
        'WOA': whale_optimization_algorithm,
        'DA': dragonfly_algorithm,
        'GWO': grey_wolf_optimizer,
        'ABCO': artificial_bee_colony_optimization,
        'BA': bat_algorithm,
        'PSO': particle_swarm_optimization,
        'FA': firefly_algorithm,
        'GA': genetic_algorithm,
        'ACO': ant_colony_optimization
    }

    # Optimizers names
    optimizer_names = list(optimizers.keys())

    def __init__(self, strategy: str, params: dict):
        """
        Initialize the Optimizer instance.

        Parameters:
            - strategy (str): The optimization name strategy to be used.
            - params (dict): The optimization parameters needed.
        """

        self.name = strategy.upper()
        if self.name in Optimizer.optimizer_names:
            self.strategy = Optimizer.optimizers[self.name]
        else:
            self.strategy = None
        self.params = params

    # ------------------------------ METHODS ------------------------------- #

    def optimize(self):
        """
        Optimize the given problem using the selected strategy and parameters.

        Returns:
            - best_solution, fitness_values (tuple): Numpy.ndarray with the best solution and a list of fitness values for each generation.
        """
        return self.strategy(self.params)

    # --------------------------- STATIC METHODS ---------------------------- #

    @staticmethod
    def get_optimizers():
        """
        Get the dictionary of optimizers.

        Returns:
            - optimizers (dict): The dictionary mapping optimizer names to their corresponding functions.
        """
        return Optimizer.optimizers

    @staticmethod
    def get_optimizers_names():
        """
        Get the list of optimizer names.

        Returns:
            - optimizer_names (list): The list of optimizer names.
        """
        return Optimizer.optimizers_names

    # TODO add algorithms parameters
    @staticmethod
    def get_default_optimizer_parameters(optimizer: str = None,
                                         solution_len: int = 2) -> dict:
        """
        Get default parameters for the specified optimizer.

        Parameters:
            - optimizer (str, optional): The optimizer for which to retrieve parameters. Defaults to None.
            - solution_len (int, optional): The length of the solution vector. Defaults to 2.

        Returns:
            - parameters (dict): A dictionary containing the default parameters for the specified optimizer.
        """
        parameters = {}
        optimizer_upper = optimizer.upper()

        if optimizer_upper == 'GOA':
            parameters = {
                'grasshoppers': DEFAULT_POPULATION_SIZE,
                'iterations': DEFAULT_ITERATIONS,
            }
        elif optimizer_upper == 'DA':
            parameters = {
                'size': DEFAULT_POPULATION_SIZE,
                'generations': DEFAULT_ITERATIONS,
                'min_values': [DEFAULT_LOWER_BOUND] * (solution_len),
                'max_values': [DEFAULT_UPPER_BOUND] * (solution_len),
                'binary': 's',
                'verbose': True,
            }
        elif optimizer_upper == 'GWO':
            parameters = {
                'pack_size': DEFAULT_POPULATION_SIZE,
                'iterations': DEFAULT_ITERATIONS,
            }
        elif optimizer_upper == 'WOA':
            parameters = {
                'hunting_party': DEFAULT_POPULATION_SIZE,
                'iterations': DEFAULT_ITERATIONS,
                'spiral_param': 1,
            }
        elif optimizer_upper == 'ABCO':
            parameters = {
                'food_sources': DEFAULT_POPULATION_SIZE,
                'iterations': DEFAULT_ITERATIONS,
                'employed_bees': 3,
                'outlookers_bees': 3,
                'limit': 3,
            }
        elif optimizer_upper == 'BA':
            parameters = {
                'swarm_size': DEFAULT_POPULATION_SIZE,
                'iterations': DEFAULT_ITERATIONS,
                'alpha': 0.9,
                'gama': 0.9,
                'fmin': 0,
                'fmax': 10,
            }
        elif optimizer_upper == 'PSO':
            parameters = {
                'swarm_size': DEFAULT_POPULATION_SIZE,
                'iterations': DEFAULT_ITERATIONS,
                'decay': 0,
                'w': 0.9,
                'c1': 2,
                'c2': 2,
            }
        elif optimizer_upper == 'FA':
            parameters = {
                'swarm_size': DEFAULT_POPULATION_SIZE,
                'generations': DEFAULT_ITERATIONS,
                'alpha_0': 0.02,
                'beta_0': 0.1,
                'gama': 1,
            }
        elif optimizer_upper == 'GA':
            parameters = {
                'population_size': DEFAULT_POPULATION_SIZE,
                'generations': DEFAULT_ITERATIONS,
                'crossover_rate': 1,
                'mutation_rate': 0.05,
                'elite': 3,
            }
        elif optimizer_upper == 'ACO':  # TODO Add parameters
            parameters = {
                'n_ants': DEFAULT_POPULATION_SIZE,
                'iterations': DEFAULT_ITERATIONS,
                'n_features': solution_len,
                'alpha': 1,
                'q': 1,
                'initial_pheromone': 0.1,
                'evaporation_rate': 0.049,  # Paper based value
            }

        parameters['verbose'] = True
        parameters['binary'] = 's'
        parameters['min_values'] = [DEFAULT_LOWER_BOUND] * (solution_len)
        parameters['max_values'] = [DEFAULT_UPPER_BOUND] * (solution_len)
        parameters['target_function_parameters'] = {
            'weights':
            np.random.uniform(low=DEFAULT_LOWER_BOUND,
                              high=DEFAULT_UPPER_BOUND,
                              size=solution_len),
            'data':
            None,
            'alpha':
            0.5,
            'classifier':
            SVC_CLASSIFIER,
            'n_neighbors':
            DEFAULT_NEIGHBORS,
            'c':
            0.1
        }

        return parameters
