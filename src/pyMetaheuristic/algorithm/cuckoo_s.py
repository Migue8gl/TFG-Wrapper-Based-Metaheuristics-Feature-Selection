############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Cuckoo Search

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import random

import numpy as np

############################################################################


# Function
def target_function():
    return


############################################################################

# Transfer functions S-Shaped


def s_shaped_transfer_function(x):
    threshold = np.random.random()
    return 1 if sigmoid(x) > threshold else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


############################################################################

# Transfer functions V-Shaped


def v_shaped_transfer_function(x):
    threshold = np.random.random()
    return 1 - x if hyperbolic_tan(x) > threshold else x


def hyperbolic_tan(x):
    return np.abs(np.tanh(x))


############################################################################


# Function: Initialize Variables
def initial_position(birds=3,
                     min_values=[-5, -5],
                     max_values=[5, 5],
                     target_function=target_function,
                     target_function_parameters=None):
    position = np.zeros((birds, len(min_values) + 4))
    for i in range(0, birds):
        for j in range(0, len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        target_function_parameters['weights'] = position[i, :-4]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['fitness']
        position[i, -2] = fitness['accuracy']
        position[i, -3] = fitness['selected_features']
        position[i, -4] = fitness['selected_rate']
    return position


############################################################################


# Function: Levy Distribution
def levy_flight(mean):
    u1 = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
    u2 = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
    v = np.random.uniform(0.0, 1.0)
    x1 = np.sin((mean - 1.0) * u1) / np.power(np.cos(u1), 1.0 / (mean - 1.0))
    x2 = np.power(
        np.cos((2.0 - mean) * u2) / (-np.log(v)), (2.0 - mean) / (mean - 1.0))
    return x1 * x2


# Function: Replace Bird
def replace_bird(position,
                 alpha_value=0.01,
                 lambda_value=1.5,
                 min_values=[-5, -5],
                 max_values=[5, 5],
                 target_function=target_function,
                 target_function_parameters=None,
                 binary='s'):
    random_bird = np.random.randint(position.shape[0])
    levy_values = levy_flight(lambda_value)
    new_solution = np.copy(position[random_bird, :-4])
    rand_factors = np.random.rand(len(min_values))
    new_solution = np.clip(
        new_solution + alpha_value * levy_values * new_solution * rand_factors,
        min_values, max_values)
    if binary != 'x':
        for j in range(0, len(min_values)):
            if binary == 's':
                new_solution[j] = s_shaped_transfer_function(new_solution[j])
            elif binary == 'v':
                new_solution[j] = v_shaped_transfer_function(new_solution[j])
    target_function_parameters['weights'] = new_solution[:]
    fitness = target_function(**target_function_parameters)

    new_solution = np.append(new_solution, fitness['selected_rate'])
    new_solution = np.append(new_solution, fitness['selected_features'])
    new_solution = np.append(new_solution, fitness['accuracy'])
    new_solution = np.append(new_solution, fitness['fitness'])
    if (fitness['fitness'] < position[random_bird, -1]):
        position[random_bird, :-4] = new_solution[:-4]
        position[random_bird, -1] = new_solution[-1]
        position[random_bird, -2] = new_solution[-2]
        position[random_bird, -3] = new_solution[-3]
        position[random_bird, -4] = new_solution[-4]
    return position


# Function: Update Positions
def update_positions(position,
                     discovery_rate=0.25,
                     min_values=[-5, -5],
                     max_values=[5, 5],
                     target_function=target_function,
                     target_function_parameters=None):
    updated_position = np.copy(position)
    abandoned_nests = int(np.ceil(discovery_rate * position.shape[0])) + 1
    fitness_values = position[:, -1]
    nest_list = np.argsort(fitness_values)[-abandoned_nests:]
    random_birds = np.random.choice(position.shape[0], size=2, replace=False)
    bird_j, bird_k = random_birds
    for i in nest_list:
        rand = np.random.rand(updated_position.shape[1] - 4)
        if np.random.rand() > discovery_rate:
            updated_position[i, :-4] = np.clip(
                updated_position[i, :-4] + rand *
                (updated_position[bird_j, :-4] -
                 updated_position[bird_k, :-4]), min_values, max_values)
        target_function_parameters['weights'] = updated_position[i, :-4]
        fitness = target_function(**target_function_parameters)
        updated_position[i, -1] = fitness['fitness']
        updated_position[i, -2] = fitness['accuracy']
        updated_position[i, -3] = fitness['selected_features']
        updated_position[i, -4] = fitness['selected_rate']
    return updated_position


############################################################################


# CS Function
def cuckoo_search(birds=3,
                  discovery_rate=0.25,
                  alpha_value=0.01,
                  lambda_value=1.5,
                  min_values=[-5, -5],
                  max_values=[5, 5],
                  iterations=50,
                  target_function=target_function,
                  target_function_parameters=None,
                  binary='s',
                  verbose=True):
    count = 0
    fitness_values = []
    position = initial_position(birds, min_values, max_values, target_function,
                                target_function_parameters)
    best_ind = np.copy(position[position[:, -1].argsort()][0, :])
    while (count <= iterations):
        if (verbose):
            print('Iteration = ', count, ' f(x) = ', best_ind[-1])
        for _ in range(0, position.shape[0]):
            position = replace_bird(position, alpha_value, lambda_value,
                                    min_values, max_values, target_function,
                                    target_function_parameters, binary)
        position = update_positions(position, discovery_rate, min_values,
                                    max_values, target_function,
                                    target_function_parameters)

        value = np.copy(position[position[:, -1].argsort()][0, :])
        if (best_ind[-1] > value[-1]):
            best_ind = np.copy(position[position[:, -1].argsort()][0, :])
        fitness_values.append({
            'fitness': best_ind[-1],
            'accuracy': best_ind[-2],
            'selected_features': best_ind[-3],
            'selected_rate': best_ind[-4]
        })
        count = count + 1
    return best_ind, fitness_values


############################################################################
