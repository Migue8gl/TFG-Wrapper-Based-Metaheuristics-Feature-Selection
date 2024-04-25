from math import sqrt

import numpy as np
from constants import (
    ALPHA,
    DATA,
    DEFAULT_ITERATIONS,
    DEFAULT_LOWER_BOUND,
    DEFAULT_NEIGHBORS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_UPPER_BOUND,
    KNN_CLASSIFIER,  # noqa: F401
    LABELS,
    SAMPLE,
    SVC_CLASSIFIER,  # noqa: F401
)
from pyMetaheuristic.algorithm import (
    ant_colony_optimization,
    artificial_bee_colony_optimization,
    bat_algorithm,
    cuckoo_search,
    differential_evolution,
    dragonfly_algorithm,
    dummy_optimizer,
    firefly_algorithm,
    genetic_algorithm,
    grasshopper_optimization_algorithm,
    grey_wolf_optimizer,
    particle_swarm_optimization,
    whale_optimization_algorithm,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class Optimizer:
    """
    Optimizer class using the Strategy design pattern.
    """

    # ----------------------------- ATTRIBUTES ------------------------------- #

    # Optimizers dict with all available optimizers
    optimizers = {
        'goa': grasshopper_optimization_algorithm,
        'woa': whale_optimization_algorithm,
        'da': dragonfly_algorithm,
        'gwo': grey_wolf_optimizer,
        'abco': artificial_bee_colony_optimization,
        'ba': bat_algorithm,
        'pso': particle_swarm_optimization,
        'fa': firefly_algorithm,
        'ga': genetic_algorithm,
        'aco': ant_colony_optimization,
        'cs': cuckoo_search,
        'de': differential_evolution,
        'dummy': dummy_optimizer
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

        self.name = strategy.lower()
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
        self.params["target_function_parameters"][DATA] = problem
        return self.strategy(**self.params)

    # --------------------------- STATIC METHODS ---------------------------- #

    @staticmethod
    def fitness(
        weights: np.ndarray,
        data: dict,
        classifier_parameters: dict,
        alpha: float = 0.99,
        classifier: str = "knn",
    ) -> dict:
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
        weights[weights <= 0.05] = 0  # Set weights less than 0.1 to 0
        selection_count = np.sum(weights != 0)
        classification_rate = Optimizer.compute_accuracy(
            weights,
            data=data,
            classifier=classifier,
            classifier_parameters=classifier_parameters,
        )
        selection_rate = selection_count / len(weights)

        # Compute fitness as a combination of classification and reduction errors
        classification_error = 1 - classification_rate
        fitness = alpha * classification_error + (1 - alpha) * selection_rate

        return {
            'fitness': fitness,
            'accuracy': classification_rate,
            'selected_features': selection_count,
            'selected_rate': selection_rate
        }

    @staticmethod
    def compute_accuracy(
        weights: np.ndarray,
        data: dict,
        classifier_parameters: dict,
        classifier: str = "knn",
    ) -> dict:
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

        if classifier == "knn":
            classifier = KNeighborsClassifier(
                n_neighbors=classifier_parameters["n_neighbors"],
                weights=classifier_parameters["weights"],
            )
        elif classifier == "svc":
            classifier = SVC(C=classifier_parameters["c"],
                             kernel=classifier_parameters["kernel"])
        else:
            classifier = KNeighborsClassifier(
                n_neighbors=classifier_parameters["n_neighbors"],
                weights=classifier_parameters["weights"],
            )

        # Train the classifier
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

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
        return Optimizer.optimizer_names

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
        optimizer_lower = optimizer.lower()

        if optimizer_lower == "goa":
            parameters = {
                "grasshoppers": DEFAULT_POPULATION_SIZE,
                "iterations": DEFAULT_ITERATIONS,
                "c_min": 0.00001,
                "c_max": 1,
                "F": 0.5,
                "L": 1.5,
                "binary": "s",
            }
        elif optimizer_lower == "da":
            parameters = {
                "size": DEFAULT_POPULATION_SIZE,
                "generations": DEFAULT_ITERATIONS,
                "binary": "s",
                "verbose": True,
            }
        elif optimizer_lower == "gwo":
            parameters = {
                "pack_size": DEFAULT_POPULATION_SIZE,
                "iterations": DEFAULT_ITERATIONS,
                "binary": "s",
            }
        elif optimizer_lower == "woa":
            parameters = {
                "hunting_party": DEFAULT_POPULATION_SIZE,
                "iterations": DEFAULT_ITERATIONS,
                "spiral_param": 1,
                "binary": "s",
            }
        elif optimizer_lower == "abco":
            parameters = {
                "food_sources": DEFAULT_POPULATION_SIZE,
                "iterations": DEFAULT_ITERATIONS,
                "employed_bees": 3,
                "outlookers_bees": 3,
                "limit": 3,
                "binary": "s",
            }
        elif optimizer_lower == "ba":
            parameters = {
                "swarm_size": DEFAULT_POPULATION_SIZE,
                "iterations": DEFAULT_ITERATIONS,
                "alpha": 0.9,
                "gama": 0.9,
                "fmin": 0,
                "fmax": 10,
                "binary": "s",
            }
        elif optimizer_lower == "pso":
            parameters = {
                "swarm_size": DEFAULT_POPULATION_SIZE,
                "iterations": DEFAULT_ITERATIONS,
                "decay": 0,
                "w": 0.9,
                "c1": 2,
                "c2": 2,
                "binary": "s",
            }
        elif optimizer_lower == "fa":
            parameters = {
                "swarm_size": DEFAULT_POPULATION_SIZE,
                "generations": DEFAULT_ITERATIONS,
                "alpha_0": 0.02,
                "beta_0": 0.1,
                "gama": 1,
                "binary": "s",
            }
        elif optimizer_lower == "ga":
            parameters = {
                "population_size": DEFAULT_POPULATION_SIZE,
                "generations": DEFAULT_ITERATIONS,
                "crossover_rate": 1,
                "mutation_rate": 0.05,
                "elite": 2,
                "eta": 1,
                "alpha": sqrt(0.3),
                'binary': True,
            }
        elif optimizer_lower == "aco":  # TODO Add parameters
            parameters = {
                "n_ants": DEFAULT_POPULATION_SIZE,
                "iterations": DEFAULT_ITERATIONS,
                "n_features": solution_len,
                "alpha": 1,
                "q": 1,
                "initial_pheromone": 0.1,
                "evaporation_rate": 0.049,  # Paper based value
            }
        elif optimizer_lower == "cs":
            parameters = {
                "birds": DEFAULT_POPULATION_SIZE,
                "iterations": DEFAULT_ITERATIONS,
                "discovery_rate": 0.25,
                "alpha_value": 1,
                "lambda_value": 1.5,
                'binary': 's',
            }
        elif optimizer_lower == "de":  # TODO Add parameters
            parameters = {
                "n": DEFAULT_POPULATION_SIZE,
                "iterations": DEFAULT_ITERATIONS,
                "F": 0.9,
                "Cr": 0.2,
                'binary': 's',
            }
        elif optimizer_lower == "dummy":  # TODO Add parameters
            parameters = {
                "swarm_size": DEFAULT_POPULATION_SIZE,
                "iterations": DEFAULT_ITERATIONS,
                'binary': 's',
            }

        parameters["verbose"] = True
        parameters["min_values"] = [DEFAULT_LOWER_BOUND] * (solution_len)
        parameters["max_values"] = [DEFAULT_UPPER_BOUND] * (solution_len)
        parameters["target_function"] = Optimizer.fitness
        parameters["target_function_parameters"] = {
            "weights":
            np.random.uniform(low=DEFAULT_LOWER_BOUND,
                              high=DEFAULT_UPPER_BOUND,
                              size=solution_len),
            "data":
            None,
            "alpha":
            ALPHA,
            "classifier":
            SVC_CLASSIFIER,
            "classifier_parameters": {
                "n_neighbors": DEFAULT_NEIGHBORS,
                "weights": "uniform",
                "c": 1,
                "kernel": "rbf",
            },
        }

        return parameters
