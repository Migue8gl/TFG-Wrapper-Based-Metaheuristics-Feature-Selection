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


def s_shaped_transfer_function(x, is_x_vector=False):
    if is_x_vector:
        threshold = np.random.random(x.shape)
        return np.where(sigmoid(x) > threshold, 1, 0)
    else:
        threshold = np.random.rand()
        return 1 if sigmoid(x) > threshold else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


############################################################################

# Transfer functions V-Shaped


def v_shaped_transfer_function(x, is_x_vector=False):
    if is_x_vector:
        threshold = np.random.random(x.shape)
        return np.where(hyperbolic_tan(x) > threshold, 1 - x, x)
    else:
        threshold = np.random.rand()
        return 1 - x if hyperbolic_tan(x) > threshold else x


def hyperbolic_tan(x):
    return np.abs(np.tanh(x))


# Function: Initialize Variables
def initial_position(swarm_size=3,
                     min_values=[-5, -5],
                     max_values=[5, 5],
                     target_function=target_function,
                     target_function_parameters=None):
    position = np.zeros((swarm_size, len(min_values) + 4))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        target_function_parameters['weights'] = position[i,
                                                         0:position.shape[1] -
                                                         4]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['fitness']
        position[i, -2] = fitness['accuracy']
        position[i, -3] = fitness['selected_features']
        position[i, -4] = fitness['selected_rate']

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
    position_[:, -4:] = 1
    beta = np.random.rand(position.shape[0])
    rand = np.random.rand(position.shape[0])
    rand_position_update = np.random.rand(position.shape[0])
    frequency[:, 0] = fmin + (fmax - fmin) * beta
    velocity = np.clip(velocity + (position[:, :-4] - best_ind[:-4]) * frequency, min_values[0], max_values[0])

    if binary == 's':
        position_[:, :-4] = s_shaped_transfer_function(velocity[:, :],
                                                       is_x_vector=True)
    elif binary == 'v':
        position_[:, :-4] = v_shaped_transfer_function(velocity[:, :],
                                                       is_x_vector=True)
    else:
        position_[:, :-4] = np.clip(position[:, :-4] + velocity, min_values,
                                    max_values)
    for i in range(0, position.shape[0]):
        target_function_parameters['weights'] = position_[
            i, 0:position_.shape[1] - 4]
        fitness = target_function(**target_function_parameters)
        position_[i, -1] = fitness['fitness']
        position_[i, -2] = fitness['accuracy']
        position_[i, -3] = fitness['selected_features']
        position_[i, -4] = fitness['selected_rate']
        if (rand[i] > rate[i, 0]):
            loudness_mean = loudness.mean()
            random_shift = np.random.uniform(-1, 1, dim) * loudness_mean
            position_[i, :-4] = np.clip(best_ind[:-4] + random_shift,
                                        min_values, max_values)
            target_function_parameters['weights'] = position_[
                i, 0:position_.shape[1] - 4]
            fitness = target_function(**target_function_parameters)
            position_[i, -1] = fitness['fitness']
            position_[i, -2] = fitness['accuracy']
            position_[i, -3] = fitness['selected_features']
            position_[i, -4] = fitness['selected_rate']
        else:
            position_[i, :] = initial_position(1, min_values, max_values,
                                               target_function,
                                               target_function_parameters)[0]
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
            'fitness': best_ind[-1],
            'accuracy': best_ind[-2],
            'selected_features': best_ind[-3],
            'selected_rate': best_ind[-4]
        })
    return best_ind, fitness_values
