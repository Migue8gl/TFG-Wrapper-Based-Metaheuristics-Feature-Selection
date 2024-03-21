# ---------------------------------- DATASETS ------------------------------------ #

D1 = './datasets/spectf-heart.arff'
D2 = './datasets/ionosphere.arff'
D3 = './datasets/parkinsons.arff'
D4 = './datasets/iris.arff'
D5 = './datasets/wine.arff'
D6 = './datasets/ecoli.arff'
D7 = './datasets/yeast.arff'
D8 = './datasets/breast-cancer.arff'
D9 = './datasets/zoo.arff'
D10 = './datasets/dermatology.arff'
D11 = './datasets/sonar.arff'
D11 = './datasets/diabetes.arff'

# ----------------------------------- NAMES ------------------------------------- #

SAMPLE = 'sample'
LABELS = 'labels'
DATA = 'data'
KNN_CLASSIFIER = 'knn'
SVC_CLASSIFIER = 'svc'
PLOT_TITLE = 'Metaheuristic optimization analysis visualization'

# ---------------------------------- DEFAULT ------------------------------------ #

ALPHA = 0.99
DEFAULT_OPTIMIZER = 'ACO'
DEFAULT_TEST_ITERATIONS = 50  # Testing purposes, can be changed
DEFAULT_ITERATIONS = 200  # Analisys iterations for each optimizer
DEFAULT_MAX_ITERATIONS = 30  # For analisys comparison between optimizers
DEFAULT_POPULATION_SIZE = 10  # Vector solution length
DEFAULT_FOLDS = 5  # Number of folds for k-fold cross validation
DEFAULT_NEIGHBORS = 5  # Number of neighbors for KNN
DEFAULT_LOWER_BOUND = 0  # Min value for features in solution
DEFAULT_UPPER_BOUND = 1  # Max value for features in solution
