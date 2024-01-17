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
    'WAO': whale_optimization_algorithm,
    'DA': dragonfly_algorithm,
    'GWO': grey_wolf_optimizer,
}

# ----------------------------------- NAMES ------------------------------------- #

DATA = 'data'
LABELS = 'labels'
KNN = 'knn'
SVC = 'svc'

# ---------------------------------- DEFAULT ------------------------------------ #

DEFAULT_OPTIMIZER = 'GOA'
