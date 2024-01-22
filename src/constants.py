from pyMetaheuristic.algorithm import grasshopper_optimization_algorithm
from pyMetaheuristic.algorithm import dragonfly_algorithm
from pyMetaheuristic.algorithm import grey_wolf_optimizer
from pyMetaheuristic.algorithm import whale_optimization_algorithm
from pyMetaheuristic.algorithm import artificial_bee_colony_optimization
from pyMetaheuristic.algorithm import bat_algorithm
from pyMetaheuristic.algorithm import bat_algorithm
from pyMetaheuristic.algorithm import particle_swarm_optimization


# ---------------------------------- DATASETS ------------------------------------ #

D1 = './datasets/spectf-heart.arff'
D2 = './datasets/ionosphere.arff'
D3 = './datasets/parkinsons.arff'

# --------------------------------- OPTIMIZERS ----------------------------------- #

OPTIMIZERS = {
    'GOA': grasshopper_optimization_algorithm,
    'WOA': whale_optimization_algorithm,
    'DA': dragonfly_algorithm,
    'GWO': grey_wolf_optimizer,
    'ABCO': artificial_bee_colony_optimization,
    'BA': bat_algorithm,
    'PSO': particle_swarm_optimization

}

# ----------------------------------- NAMES ------------------------------------- #

DATA = 'data'
LABELS = 'labels'
KNN_CLASSIFIER = 'knn'
SVC_CLASSIFIER = 'svc'
PLOT_TITLE = 'Metaheuristic optimization analysis visualization'

# ---------------------------------- DEFAULT ------------------------------------ #

DEFAULT_OPTIMIZER = 'GOA'
DEFAULT_TEST_ITERATIONS = 50+200
DEFAULT_ITERATIONS = 500
DEFAULT_MAX_ITERATIONS = 30  # For analisys comparison between optimizers
DEFAULT_POPULATION_SIZE = 20
