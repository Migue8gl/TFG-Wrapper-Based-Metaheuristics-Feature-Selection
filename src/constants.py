# ---------------------------------- DATASETS ------------------------------------ #

D1 = './datasets/spectf-heart.arff'
D2 = './datasets/ionosphere.arff'
D3 = './datasets/parkinsons.arff'

# ----------------------------------- NAMES ------------------------------------- #

SAMPLE = 'sample'
LABELS = 'labels'
DATA = 'data'
KNN_CLASSIFIER = 'knn'
SVC_CLASSIFIER = 'svc'
PLOT_TITLE = 'Metaheuristic optimization analysis visualization'

# ---------------------------------- DEFAULT ------------------------------------ #

DEFAULT_OPTIMIZER = 'ACO'
DEFAULT_TEST_ITERATIONS = 100  # Testing purposes, can be changed
DEFAULT_ITERATIONS = 500  # Analisys iterations for each optimizer
DEFAULT_MAX_ITERATIONS = 30  # For analisys comparison between optimizers
DEFAULT_POPULATION_SIZE = 20  # Vector solution length
DEFAULT_FOLDS = 5  # Number of folds for k-fold cross validation
DEFAULT_NEIGHBORS = 10  # Number of neighbors for KNN
DEFAULT_LOWER_BOUND = 0  # Min value for features in solution
DEFAULT_UPPER_BOUND = 1  # Max value for features in solution
