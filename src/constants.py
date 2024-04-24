# ---------------------------------- data ------------------------------------ #

D1 = 'data/spectf-heart.arff'
D2 = 'data/ionosphere.arff'
D3 = 'data/parkinsons.arff'
D4 = 'data/iris.arff'
D5 = 'data/wine.arff'
D6 = 'data/ecoli.arff'
D7 = 'data/yeast.arff'
D8 = 'data/breast-cancer.arff'
D9 = 'data/zoo.arff'
D10 = 'data/dermatology.arff'
D11 = 'data/sonar.arff'
D11 = 'data/diabetes.arff'
D12 = 'data/wdbc.arff'

# ----------------------------------- NAMES ------------------------------------- #

SAMPLE = 'sample'
LABELS = 'labels'
DATA = 'data'
KNN_CLASSIFIER = 'knn'
SVC_CLASSIFIER = 'svc'
PLOT_TITLE = 'Metaheuristic optimization analysis visualization'

# -------------------------------- DIRECTORIES ---------------------------------- #

IMG_DIR = 'img/'
RESULTS_DIR = 'results/'
CREDENTIALS_DIR = 'creds/'

# ---------------------------------- DEFAULT ------------------------------------ #

ALPHA = 0.9
DEFAULT_OPTIMIZER = 'ACO'
DEFAULT_TEST_ITERATIONS = 50  # Testing purposes, can be changed
DEFAULT_ITERATIONS = 200  # Analisys iterations for each optimizer
DEFAULT_MAX_ITERATIONS = 30  # For analisys comparison between optimizers
DEFAULT_POPULATION_SIZE = 10  # Vector solution length
DEFAULT_FOLDS = 5  # Number of folds for k-fold cross validation
DEFAULT_NEIGHBORS = 5  # Number of neighbors for KNN
DEFAULT_LOWER_BOUND = 0  # Min value for features in solution
DEFAULT_UPPER_BOUND = 1  # Max value for features in solution
