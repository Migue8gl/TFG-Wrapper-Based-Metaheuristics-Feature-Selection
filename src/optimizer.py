from pyMetaheuristic.algorithm import (
    grasshopper_optimization_algorithm, dragonfly_algorithm,
    grey_wolf_optimizer, whale_optimization_algorithm,
    artificial_bee_colony_optimization, bat_algorithm, firefly_algorithm,
    particle_swarm_optimization, genetic_algorithm, ant_colony_optimization)
from constants import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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

    def optimize(self, problem: dict) -> tuple:
        """
        Optimize the given problem using the selected strategy and parameters.

        Parameters:
            - problem (dict): The problem to be optimized in dict form. It keys must be 'samples' and 'labels'.

        Returns:
            - best_solution, fitness_values (tuple): Numpy.ndarray with the best solution and a list of fitness values for each generation.
        """
        self.params['target_function_parameters'][DATA] = problem
        return self.strategy(**self.params)

    # --------------------------- STATIC METHODS ---------------------------- #

    @staticmethod
    def fitness(weights: np.ndarray,
                data: dict,
                classifier_parameters: dict,
                alpha: float = 0.5,
                classifier: str = 'knn') -> dict:
        """
        Functionality to compute the accuracy using KNN or SVC classifier

        Parameters:
            - weights (np.ndarray): The weights to be used for each feature.
            - data (dict): The dataset in dict form splitted into samples and labels.
            - classifier_parameters (dict): The classifier parameters. 
            - alpha (float, optional): The alpha parameter for combining classification and reduction errors. Defaults to 0.5.
            - classifier (str, optional): The classifier to be used. Defaults to 'knn'.

        Returns:
            - fitness (dict): The dictionary containing training fitness and validation fitness.
        """
        # Count number of features with zero importance.
        reduction_count = np.sum(weights == 0)
        classification_rate = Optimizer.compute_accuracy(
            weights,
            data=data,
            classifier=classifier,
            classifier_parameters=classifier_parameters)
        reduction_rate = reduction_count / len(weights)

        # Calculate the error rates in training
        classification_error = 1 - classification_rate['TrainError']
        reduction_error = 1 - reduction_rate

        # Compute fitness as a combination of classification and reduction errors
        fitness_train = alpha * classification_error + \
            (1 - alpha) * reduction_error
        classification_error = 1 - classification_rate['ValError']
        fitness_val = alpha * classification_error + (1 -
                                                      alpha) * reduction_error

        return {'TrainFitness': fitness_train, 'ValFitness': fitness_val}

    @staticmethod
    def compute_accuracy(weights: np.ndarray,
                         data: dict,
                         classifier_parameters: dict,
                         classifier: str = 'knn') -> dict:
        """
        Functionality to compute the accuracy using KNN or SVC classifier

        Parameters:
            - weights (np.ndarray): The weights to be used for each feature.
            - data (dict): The dataset in dict form splitted into samples and labels.
            - classifier_parameters (dict): The classifier parameters. 
            - classifier (str, optional): The classifier to be used.

        Returns:
            - errors (dict): The dictionary containing e_in error and e_out error.
        """
        sample = data[SAMPLE]
        labels = data[LABELS]

        # Giving each characteristic an importance by multiplying the sample and weights
        sample_weighted = np.multiply(sample, weights)
        # Split into train and test data
        x_train, x_test, y_train, y_test = train_test_split(sample_weighted,
                                                            labels,
                                                            test_size=0.2,
                                                            random_state=42)

        if (classifier == 'knn'):
            classifier = KNeighborsClassifier(
                n_neighbors=classifier_parameters['n_neighbors'],
                weights=classifier_parameters['weights'])
        elif (classifier == 'svc'):
            classifier = SVC(C=classifier_parameters['C'],
                             kernel=classifier_parameters['kernel'])
        else:
            print('No valid classifier, using KNN by default')
            classifier = KNeighborsClassifier(
                n_neighbors=classifier_parameters['n_neighbors'],
                weights=classifier_parameters['weights'])

        # Train the classifier
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_train)
        e_in = accuracy_score(y_train, y_pred)

        y_pred = classifier.predict(x_test)
        e_out = accuracy_score(y_test, y_pred)

        return {'TrainError': e_in, 'ValError': e_out}

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
        parameters['target_function'] = Optimizer.fitness
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
            'classifier_parameters': {
                'n_neighbors': DEFAULT_NEIGHBORS,
                'weights': 'distance',
                'C': 1,
                'kernel': 'rbf'
            }
        }

        return parameters
