from pyMetaheuristic.algorithm import grasshopper_optimization_algorithm
from pyMetaheuristic.algorithm import dragonfly_algorithm
from pyMetaheuristic.algorithm import grey_wolf_optimizer
from pyMetaheuristic.algorithm import whale_optimization_algorithm

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
}

# ----------------------------------- NAMES ------------------------------------- #

DATA = 'data'
LABELS = 'labels'
KNN_CLASSIFIER = 'knn'
SVC_CLASSIFIER = 'svc'

# ---------------------------------- DEFAULT ------------------------------------ #

DEFAULT_OPTIMIZER = 'GOA'
DEFAULT_TEST_ITERATIONS = 50
DEFAULT_ITERATIONS = 2
DEFAULT_MAX_ITERATIONS = 30 # For analisys comparison between optimizers
DEFAULT_POPULATION_SIZE = 20
