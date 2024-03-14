# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Bat Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

# Required Libraries
import random

import numpy as np


# Function
def target_function():
    return


# Transfer functions S-Shaped
def s_shaped_transfer_function(x):
    threshold = np.random.random()
    return 1 if sigmoid(x) > threshold else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Transfer functions V-Shaped
def v_shaped_transfer_function(x):
    threshold = np.random.random()
    return 1 - x if hyperbolic_tan(x) > threshold else x


def hyperbolic_tan(x):
    return np.abs(np.tanh(x))


# Function: Initialize Variables
def initial_position(swarm_size=3,
                     min_values=[-5, -5],
                     max_values=[5, 5],
                     target_function=target_function,
                     target_function_parameters=None):
    position = np.zeros((swarm_size, len(min_values) + 2))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        target_function_parameters['weights'] = position[i,
                                                         0:position.shape[1] -
                                                         2]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['validation']['fitness']
        position[i, -2] = fitness['training']['fitness']

    return position


# Function: Initialize Variables
def initial_variables(swarm_size, dim):
    velocity = np.zeros((swarm_size, dim))
    frequency = np.zeros((swarm_size, 1))
    rate = np.random.rand(swarm_size, 1)
    loudness = np.random.uniform(1, 2, (swarm_size, 1))
    return velocity, frequency, rate, loudness


# Function: Update Position
def update_position(position,
                    velocity,
                    frequency,
                    rate,
                    loudness,
                    best_ind,
                    alpha,
                    gama,
                    fmin,
                    fmax,
                    count,
                    min_values,
                    max_values,
                    target_function=target_function,
                    target_function_parameters=None,
                    binary='s'):
    dim = len(min_values)
    position_ = np.zeros_like(position)
    beta = np.random.rand(position.shape[0])
    rand = np.random.rand(position.shape[0])
    rand_position_update = np.random.rand(position.shape[0])
    frequency[:, 0] = fmin + (fmax - fmin) * beta
    velocity = velocity + (position[:, :-2] - best_ind[:-2]) * frequency

    for i in range(len(position_)):
        for k in range(len(max_values)):
            if binary == 's':
                position_[i, k] = s_shaped_transfer_function(velocity[i, k])
            elif binary == 'v':
                position_[i, k] = v_shaped_transfer_function(velocity[i, k])
            else:
                position_[i, k] = position[i, k] + velocity[i, k]
    for i in range(0, position.shape[0]):
        target_function_parameters['weights'] = position_[
            i, 0:position_.shape[1] - 2]
        fitness = target_function(**target_function_parameters)
        position_[i, -1] = fitness['validation']['fitness']
        position_[i, -2] = fitness['training']['fitness']
        if (rand[i] > rate[i, 0]):
            loudness_mean = loudness.mean()
            random_shift = np.random.uniform(-1, 1, dim) * loudness_mean
            position_[i, :-2] = np.clip(best_ind[:-2] + random_shift,
                                        min_values, max_values)
            target_function_parameters['weights'] = position_[
                i, 0:position_.shape[1] - 2]
            fitness = target_function(**target_function_parameters)
            position_[i, -1] = fitness['validation']['fitness']
            position_[i, -2] = fitness['training']['fitness']
        else:
            position_[i, :] = initial_position(1, min_values, max_values,
                                               target_function, target_function_parameters)[0]
        if (rand_position_update[i] < loudness[i, 0]
                and position_[i, -1] <= position[i, -1]):
            position[i, :] = position_[i, :]
            rate[i, 0] = np.random.rand() * (1 - np.exp(-gama * count))
            loudness[i, 0] = loudness[i, 0] * alpha
    position = np.vstack([position, position_])
    position = position[position[:, -1].argsort()]
    position = position[:position_.shape[0], :]
    best_index = np.argmin(position[:, -1])
    if (best_ind[-1] > position[best_index, -1]):
        best_ind = np.copy(position[best_index, :])
    return position, velocity, frequency, rate, loudness, best_ind


# BA Function
def bat_algorithm(swarm_size=3,
                  min_values=[-5, -5],
                  max_values=[5, 5],
                  iterations=50,
                  alpha=0.9,
                  gama=0.9,
                  fmin=0,
                  fmax=10,
                  target_function=target_function,
                  target_function_parameters=None,
                  binary='s',
                  verbose=True):
    count = 0
    fitness_values = []
    position = initial_position(swarm_size, min_values, max_values,
                                target_function, target_function_parameters)
    velocity, frequency, rate, loudness = initial_variables(
        swarm_size, len(min_values))
    best_ind = np.copy(position[position[:, -1].argsort()][0, :])
    while (count <= iterations):
        if (verbose):
            print('Iteration = ', count, ' f(x) = ', best_ind[-1])
        position, velocity, frequency, rate, loudness, best_ind = update_position(
            position, velocity, frequency, rate, loudness, best_ind, alpha,
            gama, fmin, fmax, count, min_values, max_values, target_function,
            target_function_parameters, binary)
        count = count + 1
        fitness_values.append({
            'val_fitness': best_ind[-1],
            'train_fitness': best_ind[-2]
        })
    return best_ind, fitness_values
