############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Firefly Algorithm

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
    threshold = np.random.random(x.shape)
    return np.where(sigmoid(x) > threshold, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


############################################################################

# Transfer functions V-Shaped


def v_shaped_transfer_function(x):
    threshold = np.random.random(x.shape)
    return np.where(hyperbolic_tan(x) > threshold, 1 - x, x)


def hyperbolic_tan(x):
    return np.abs(np.tanh(x))


############################################################################


# Function: Initialize Variables
def initial_fireflies(swarm_size=3,
                      min_values=[-5, -5],
                      max_values=[5, 5],
                      target_function=target_function,
                      target_function_parameters=None):
    position = np.zeros((swarm_size, len(min_values) + 4))
    for i in range(0, swarm_size):
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


# Function: Distance Calculations
def euclidean_distance(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y))**2))


############################################################################


# Function: Beta Value
def beta_value(x, y, gama, beta_0):
    rij = euclidean_distance(x, y)
    beta = beta_0 * np.exp(-gama * (rij**2))
    return beta


# Function: Light Intensity
def light_value(light_0, x, y, gama):
    rij = euclidean_distance(x, y)
    light = light_0 * np.exp(-gama * (rij**2))
    return light


# Function: Update Position
def update_position(position, alpha_0, beta_0, gama, min_values, max_values,
                    target_function, target_function_parameters, binary):
    dim = len(min_values)
    position_ = np.copy(position)
    for i in range(position.shape[0]):
        for j in range(position.shape[0]):
            if (i != j):
                firefly_i = position[i, :-4]
                firefly_j = position[j, :-4]
                light_i = light_value(position[i, -1], firefly_i, firefly_j,
                                      gama)
                light_j = light_value(position[j, -1], firefly_i, firefly_j,
                                      gama)
                if (light_i > light_j):
                    epson = np.random.rand(dim)-0.5
                    beta = beta_value(firefly_i, firefly_j, gama, beta_0)
                    if binary == 's':
                        position[i, :-4] = s_shaped_transfer_function(
                            firefly_i + beta * (firefly_j - firefly_i) +
                            alpha_0 * epson)
                    elif binary == 'v':
                        position[i, :-4] = v_shaped_transfer_function(
                            firefly_i + beta * (firefly_j - firefly_i) +
                            alpha_0 * epson, firefly_i)
                    else:
                        position[i, :-4] = np.clip(
                            firefly_i + beta * (firefly_j - firefly_i) +
                            alpha_0 * epson, min_values, max_values)
                    target_function_parameters['weights'] = position[i, :-4]
                    fitness = target_function(**target_function_parameters)
                    position[i, -1] = fitness['fitness']
                    position[i, -2] = fitness['accuracy']
                    position[i, -3] = fitness['selected_features']
                    position[i, -4] = fitness['selected_rate']
    all_positions = np.vstack([position, position_])
    all_positions = all_positions[np.argsort(all_positions[:, -1])]
    position = all_positions[:position_.shape[0], :]
    return position


############################################################################


# Function: FFA
def firefly_algorithm(swarm_size=25,
                      min_values=[-5, -5],
                      max_values=[5, 5],
                      generations=5000,
                      alpha_0=0.2,
                      beta_0=1,
                      gama=1,
                      target_function=target_function,
                      verbose=True,
                      target_function_parameters=None,
                      binary='s'):
    position = initial_fireflies(swarm_size, min_values, max_values,
                                 target_function, target_function_parameters)
    best_firefly = np.copy(position[position[:, -1].argsort()][0, :])
    count = 0
    fitness_values = []
    while (count <= generations):
        if (verbose):
            print('Generation: ', count, ' f(x) = ', best_firefly[-1])
        position = update_position(position, alpha_0, beta_0, gama, min_values,
                                   max_values, target_function,
                                   target_function_parameters, binary)
        best_firefly = np.copy(position[position[:, -1].argsort()][0, :])
        fitness_values.append({
            'fitness': best_firefly[-1],
            'accuracy': best_firefly[-2],
            'selected_features': best_firefly[-3],
            'selected_rate': best_firefly[-4]
        })
        count += 1
    return best_firefly, fitness_values


############################################################################
