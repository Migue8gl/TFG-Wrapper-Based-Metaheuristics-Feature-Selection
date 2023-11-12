from pyMetaheuristic.algorithm import grasshopper_optimization_algorithm
import numpy as np
import arff
import matplotlib.pyplot as plt

def load_arff_data(file_path):
    try:
        with open(file_path, 'r') as arff_file:
            dataset = arff.load(arff_file)
            data = np.array(dataset['data'])
            
            # Transform all columns except the last one to float64
            data[:, :-1] = data[:, :-1].astype(np.float64)
            
        return data
    except Exception as e:
        print(f"An error occurred while loading the ARFF data: {str(e)}")
        return None

from sklearn.preprocessing import MinMaxScaler

def normalize_data(data):
    # Separate the features (X) and labels (y)
    x = data[:, :-1].astype(np.float64)
    y = data[:, -1]

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit the scaler to the features and transform the data
    x_normalized = scaler.fit_transform(x)

    # Combine the normalized features (X_normalized) and labels (y) into a single array
    normalized_data = np.column_stack((x_normalized, y))

    return normalized_data

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Read data
d1 = './datasets/spectf-heart.arff'
d2 = './datasets/ionosphere.arff'
d3 = './datasets/parkinsons.arff'
dataset = load_arff_data(d2)

# Split the data into training and testing sets
samples = dataset[:, :-1].astype(np.float64)
classes = dataset[:, -1]
x_train, x_test, y_train, y_test = train_test_split(samples, classes, test_size=0.2, random_state=42)
sample = {'data': x_train, 'labels': y_train}
sample_test = {'data': x_test, 'labels': y_test}

def compute_accuracy(weights, data=sample, classifier='knn', number_neighbors=5):
    sample = data['data']
    labels = data['labels']

    sample_weighted = np.multiply(sample, weights)

    if (classifier == 'knn'):
        classifier = KNeighborsClassifier(n_neighbors=number_neighbors, weights='uniform')
    else:
        classifier = SVC(kernel='rbf')

    # Train the classifier
    classifier.fit(sample_weighted, labels)
    y_pred = classifier.predict(sample_weighted)

    # Calculate accuracy
    accuracy = accuracy_score(labels, y_pred)

    return accuracy

def fitness(weights=None, data=sample, alpha=0.5, classifier='knn'):
    if weights is None:
        weights = np.random.uniform(low=0, high=1, size=data['data'].shape[1])

    reduction_count = np.sum(weights == 0)
    weights[weights < 0.1] = 0.0
    classification_rate = compute_accuracy(weights, data=data, classifier=classifier) 
    reduction_rate = reduction_count / len(weights)

    # Calculate the error as a percentage
    classification_error = 1 - classification_rate
    reduction_error = 1 - reduction_rate

    # Compute fitness as a combination of classification and reduction errors
    fitness = alpha * classification_error + (1 - alpha) * reduction_error

    return fitness

def plot_fitness_over_folds(fitness_values, iterations, k, ax=None):
    iteration_numbers = np.arange(0, iterations+1)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(iteration_numbers, fitness_values, label='Fitness Value')
    ax.set_title('Average Fitness Value Over {} Folds'.format(k))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness Value')
    ax.legend()

def plot_train_val_fitness(test_fitness, training_fitness, ax=None):
    x = np.arange(0, len(test_fitness))
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, training_fitness, label='Training Fitness', color='blue')
    ax.plot(x, test_fitness, label='Validation Fitness', color='orange')

    ax.set_title('Validation vs Training Fitness')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness Value')
    ax.legend()

from sklearn.model_selection import StratifiedKFold
# Function to perform k-fold cross-validation
def k_fold_cross_validation(samples, classes, k=5, parameters=None):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    test_fitness = []
    training_fitness = []
    fitness_each_fold = {}
    fold_index = 0
    average_fitness_values = []

    for train_index, test_index in skf.split(samples, classes):
        x_train, x_test = samples[train_index], samples[test_index]
        y_train, y_test = classes[train_index], classes[test_index]

        sample = {'data': x_train, 'labels': y_train}
        sample_test = {'data': x_test, 'labels': y_test}

        # Run optimization algorithm on the current fold
        gao, fitness_values = grasshopper_optimization_algorithm(target_function=fitness, **parameters)
        fitness_each_fold[fold_index] = fitness_values

        # Evaluate the model on the test set of the current fold
        test_fitness.append(fitness(weights=gao[:-1], data=sample_test))
        training_fitness.append(gao[-1])

        fold_index += 1
        print('\n##### Finished {} fold #####\n'.format(fold_index))

    # Transpose fitness values to have each list represent values for a specific index
    transposed_fitness = np.array(list(fitness_each_fold.values())).T

    # Calculate mean of each index across all lists
    average_fitness_values = np.mean(transposed_fitness, axis=1)

    return test_fitness, np.array(training_fitness), np.array(average_fitness_values)

# Optimization function's parameters
parameters = {
    'grasshoppers': 20,
    'min_values': [0] * (dataset.shape[1]-1),
    'max_values': [1] * (dataset.shape[1]-1),
    'iterations': 650,
    'binary': 's'
}

# Perform k-fold cross-validation
k = 5
test_fitness, training_fitness, k_fold_fitness_values = k_fold_cross_validation(samples, classes, k, parameters=parameters)

# Print average accuracy over k folds
print('Average Test fitness over 5 Folds: ', round(np.mean(test_fitness), 4))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
plot_fitness_over_folds(k_fold_fitness_values, parameters['iterations'], k, ax=axes[0])
plot_train_val_fitness(test_fitness, training_fitness, ax=axes[1])
plt.tight_layout()
plt.show()

# Plot Solution
"""
from pyMetaheuristic.utils import graphs
plot_parameters = {
    'min_values': (-5, -5),
    'max_values': (5, 5),
    'step': (0.1, 0.1),
    'solution': [variables],
    'proj_view': '3D',
    'view': 'browser'
}
graphs.plot_single_function(target_function = fitness, **plot_parameters)
"""