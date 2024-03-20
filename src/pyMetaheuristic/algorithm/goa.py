############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Grasshopper Optimization Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np
import random

############################################################################

# Function


def target_function():
    return


############################################################################

# Function: Initialize Variables


def initial_position(grasshoppers=5,
                     min_values=[-5, -5],
                     max_values=[5, 5],
                     target_function=target_function,
                     target_function_parameters=None):
    position = np.zeros((grasshoppers, len(min_values) + 4))
    for i in range(0, grasshoppers):
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

# Transfer functions S-Shaped


def s_shaped_transfer_function(x):
    threshold = np.random.random(x.shape)
    return np.where(sigmoid(x) > threshold, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


############################################################################

# Transfer functions V-Shaped


def v_shaped_transfer_function(x, delta_x):
    threshold = np.random.random(x.shape)
    return np.where(hyperbolic_tan(delta_x) > threshold, 1 - x, x)


def hyperbolic_tan(x):
    return np.abs(np.tanh(x))


############################################################################

# Function: S


def s_function(r, F, L):
    s = F * np.exp(-r / L) - np.exp(-r)
    return s


# Function: Distance Matrix


# Function: Distance Matrix
def build_distance_matrix(position):
    a = position[:, :-4]
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    return np.sqrt(np.einsum('ijk,ijk->ij', b - a, b - a)).squeeze()


# Function: Update Position
def update_position(position, best_position, min_values, max_values, C, F, L,
                    target_function, target_function_parameters, binary):
    dim = len(min_values)
    distance_matrix = build_distance_matrix(position)
    distance_matrix = 2 * (distance_matrix - np.min(distance_matrix)) / (
        np.ptp(distance_matrix) + 1e-8) + 1
    np.fill_diagonal(distance_matrix, 0)
    for j in range(dim):
        sum_grass = np.zeros(position.shape[0])
        for i in range(position.shape[0]):
            s_vals = s_function(distance_matrix[:, i], F, L)
            denominator = np.where(distance_matrix[:, i] == 0, 1,
                                   distance_matrix[:, i])
            sum_grass[i] = np.sum(
                C * ((max_values[j] - min_values[j]) / 2) * s_vals *
                ((position[:, j] - position[i, j]) / denominator))
        if binary == 's':
            position[:, j] = s_shaped_transfer_function(C * sum_grass)
        elif binary == 'v':
            position[:, j] = v_shaped_transfer_function(
                C * sum_grass + best_position[j], C * sum_grass)
        else:
            position[:, j] = np.clip(C * sum_grass + best_position[j],
                                     min_values[j], max_values[j])
    for i in range(position.shape[0]):
        target_function_parameters['weights'] = position[i, :-4]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['fitness']
        position[i, -2] = fitness['accuracy']
        position[i, -3] = fitness['selected_features']
        position[i, -4] = fitness['selected_rate']
    return position


############################################################################

# GOA Function


def grasshopper_optimization_algorithm(grasshoppers=5,
                                       min_values=[-5, -5],
                                       max_values=[5, 5],
                                       c_min=0.00004,
                                       c_max=1,
                                       iterations=1000,
                                       F=0.5,
                                       L=1.5,
                                       target_function=target_function,
                                       binary='x',
                                       verbose=True,
                                       target_function_parameters=None):
    count = 0
    position = initial_position(grasshoppers, min_values, max_values,
                                target_function, target_function_parameters)
    best_position = np.copy(position[np.argmin(position[:, -1]), :])

    # Lists to store fitness values
    fitness_values = []

    while (count <= iterations):
        C = c_max - count * ((c_max - c_min) / iterations)
        position = update_position(
            position,
            best_position,
            min_values,
            max_values,
            C,
            F,
            L,
            target_function=target_function,
            binary=binary,
            target_function_parameters=target_function_parameters)
        if (np.amin(position[:, -1]) < best_position[-1]):
            best_position = np.copy(position[np.argmin(position[:, -1]), :])
        count = count + 1
        if (verbose):
            print('Iteration = ', count, ' f(x) = ', best_position[-1])
        fitness_values.append({
            'fitness': best_position[-1],
            'accuracy': best_position[-2],
            'selected_features': best_position[-3],
            'selected_rate': best_position[-4]
        })
    return best_position, fitness_values


############################################################################
