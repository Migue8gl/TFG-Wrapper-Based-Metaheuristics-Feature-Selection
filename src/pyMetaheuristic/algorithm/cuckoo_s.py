############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Cuckoo Search

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np
import math
import random
import os

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


def v_shaped_transfer_function(x, delta_x):
    threshold = np.random.random()
    return 1 - x if hyperbolic_tan(delta_x) > threshold else x


def hyperbolic_tan(x):
    return np.abs(np.tanh(x))


############################################################################


# Function: Initialize Variables
def initial_position(birds=3,
                     min_values=[-5, -5],
                     max_values=[5, 5],
                     target_function=target_function,
                     target_function_parameters=None):
    position = np.zeros((birds, len(min_values) + 2))
    for i in range(0, birds):
        for j in range(0, len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        target_function_parameters['weights'] = position[i, :-2]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['validation']['fitness']
        position[i, -2] = fitness['training']['fitness']
    return position


############################################################################


# Function: Levy Distribution
def levy_flight(mean):
    x1 = math.sin((mean - 1.0) *
                  (random.uniform(-0.5 * math.pi, 0.5 * math.pi))) / (math.pow(
                      math.cos(
                          (random.uniform(-0.5 * math.pi, 0.5 * math.pi))),
                      (1.0 / (mean - 1.0))))
    x2 = math.pow((math.cos(
        (2.0 - mean) * (random.uniform(-0.5 * math.pi, 0.5 * math.pi))) /
                   (-math.log(random.uniform(0.0, 1.0)))),
                  ((2.0 - mean) / (mean - 1.0)))
    return x1 * x2


# Function: Replace Bird
def replace_bird(position,
                 alpha_value=0.01,
                 lambda_value=1.5,
                 min_values=[-5, -5],
                 max_values=[5, 5],
                 target_function=target_function,
                 target_function_parameters=None):
    random_bird = np.random.randint(position.shape[0], size=1)[0]
    new_solution = np.zeros((1, position.shape[1]))
    for j in range(0, position.shape[1] - 2):
        new_solution[0, j] = np.clip(
            position[random_bird, j] + alpha_value *
            levy_flight(lambda_value) * position[random_bird, j] *
            (int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)),
            min_values[j], max_values[j])
    target_function_parameters['weights'] = new_solution[0, :-2]
    fitness = target_function(**target_function_parameters)
    new_solution[0, -1] = fitness['validation']['fitness']
    new_solution[0, -2] = fitness['training']['fitness']
    if (position[random_bird, -1] > new_solution[0, -1]):
        position[random_bird, j] = np.copy(new_solution[0, j])
    return position


# Function: Update Positions
def update_positions(position,
                     discovery_rate=0.25,
                     min_values=[-5, -5],
                     max_values=[5, 5],
                     target_function=target_function,
                     target_function_parameters=None):
    updated_position = np.copy(position)
    abandoned_nests = math.ceil(discovery_rate * updated_position.shape[0]) + 1
    random_bird_j = np.random.randint(position.shape[0], size=1)[0]
    random_bird_k = np.random.randint(position.shape[0], size=1)[0]
    while (random_bird_j == random_bird_k):
        random_bird_j = np.random.randint(position.shape[0], size=1)[0]
    nest_list = list(position.argsort()[-(abandoned_nests - 1):][::-1][0])
    for i in range(0, updated_position.shape[0]):
        for j in range(0, len(nest_list)):
            rand = int.from_bytes(os.urandom(8), byteorder='big') / (
                (1 << 64) - 1)
            if (i == nest_list[j] and rand > discovery_rate):
                for k in range(0, updated_position.shape[1] - 2):
                    rand = int.from_bytes(os.urandom(8), byteorder='big') / (
                        (1 << 64) - 1)
                    updated_position[i, k] = np.clip(
                        updated_position[i, k] + rand *
                        (updated_position[random_bird_j, k] -
                         updated_position[random_bird_k, k]), min_values[k],
                        max_values[k])
        target_function_parameters['weights'] = updated_position[i, :-2]
        fitness = target_function(**target_function_parameters)
        updated_position[i, -1] = fitness['validation']['fitness']
        updated_position[i, -2] = fitness['training']['fitness']
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
    position = initial_position(birds, min_values, max_values, target_function)
    best_ind = np.copy(position[position[:, -1].argsort()][0, :])
    while (count <= iterations):
        if (verbose):
            print('Iteration = ', count, ' f(x) = ', best_ind[-1])
        fitness_values.append({
            'val_fitness': best_ind[-1],
            'train_fitness': best_ind[-2]
        })
        for _ in range(0, position.shape[0]):
            position = replace_bird(position, alpha_value, lambda_value,
                                    min_values, max_values, target_function)
        position = update_positions(position, discovery_rate, min_values,
                                    max_values, target_function)
        value = np.copy(position[position[:, -1].argsort()][0, :])
        if (best_ind[-1] > value[-1]):
            best_ind = np.copy(position[position[:, -1].argsort()][0, :])
        count = count + 1
    return best_ind, fitness_values


############################################################################
