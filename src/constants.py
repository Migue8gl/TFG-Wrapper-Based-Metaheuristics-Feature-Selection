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
    'gao': grasshopper_optimization_algorithm,
    'wao': whale_optimization_algorithm,
    'da': dragonfly_algorithm,
    'gwo': grey_wolf_optimizer,
}

# ----------------------------------- NAMES ------------------------------------- #

DATA = 'data'
LABELS = 'labels'
KNN = 'knn'
SVC = 'svc'

# ---------------------------------- DEFAULT ------------------------------------ #

DEFAULT_OPTIMIZER = 'gao'
