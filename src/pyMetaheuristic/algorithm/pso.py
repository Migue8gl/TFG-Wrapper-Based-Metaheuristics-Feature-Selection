############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Particle Swarm Optimization

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


def v_shaped_transfer_function(x, delta_x, is_x_vector=False):
    if is_x_vector:
        threshold = np.random.random(x.shape)
        return np.where(hyperbolic_tan(delta_x) > threshold, 1 - x, x)
    else:
        threshold = np.random.rand()
        return 1 - x if hyperbolic_tan(delta_x) > threshold else x


def hyperbolic_tan(x):
    return np.abs(np.tanh(x))


############################################################################

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
        target_function_parameters['weights'] = position[i, :-4]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['fitness']
        position[i, -2] = fitness['accuracy']
        position[i, -3] = fitness['selected_features']
        position[i, -4] = fitness['selected_rate']
    return position


############################################################################

# Function: Initialize Velocity


# Function: Initialize Velocity
def initial_velocity(position, min_values, max_values):
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    return np.random.uniform(min_values, max_values,
                             (position.shape[0], len(min_values)))


# Function: Individual Best
def individual_best_matrix(position, i_b_matrix):
    better_fitness_mask = position[:, -1] < i_b_matrix[:, -1]
    i_b_matrix[better_fitness_mask] = position[better_fitness_mask]
    return i_b_matrix


# Function: Velocity
def velocity_vector(position, init_velocity, i_b_matrix, best_global, w, c1,
                    c2):
    r1 = np.random.rand(position.shape[0], position.shape[1] - 4)
    r2 = np.random.rand(position.shape[0], position.shape[1] - 4)
    velocity = w * init_velocity + c1 * r1 * (
        i_b_matrix[:, :-4] - position[:, :-4]) + c2 * r2 * (best_global[:-4] -
                                                            position[:, :-4])
    return velocity


# Function: Updtade Position
def update_position(position,
                    velocity,
                    min_values,
                    max_values,
                    target_function,
                    target_function_parameters=None,
                    binary='s'):
    if binary == 's':
        position[:, :-4] = s_shaped_transfer_function(velocity,
                                                      is_x_vector=True)
    elif binary == 'v':
        position[:, :-4] = v_shaped_transfer_function(position[:, :-4],
                                                      velocity,
                                                      is_x_vector=True)
    else:
        position[:, :-4] = np.clip((position[:, :-4] + velocity), min_values,
                                   max_values)

    for i in range(0, position.shape[0]):
        target_function_parameters['weights'] = position[i, :-4]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['fitness']
        position[i, -2] = fitness['accuracy']
        position[i, -3] = fitness['selected_features']
        position[i, -4] = fitness['selected_rate']
    return position


############################################################################

# PSO Function


def particle_swarm_optimization(swarm_size=3,
                                min_values=[-5, -5],
                                max_values=[5, 5],
                                iterations=50,
                                decay=0,
                                w=0.9,
                                c1=2,
                                c2=2,
                                target_function=target_function,
                                verbose=True,
                                target_function_parameters=None,
                                binary='s'):
    count = 0
    position = initial_position(swarm_size, min_values, max_values,
                                target_function, target_function_parameters)
    init_velocity = initial_velocity(position, min_values, max_values)
    i_b_matrix = np.copy(position)
    best_global = np.copy(position[position[:, -1].argsort()][0, :])
    fitness_values = []
    while (count <= iterations):
        if (verbose):
            print('Iteration = ', count, ' f(x) = ', best_global[-1])
        position = update_position(position, init_velocity, min_values,
                                   max_values, target_function,
                                   target_function_parameters, binary)
        i_b_matrix = individual_best_matrix(position, i_b_matrix)
        value = np.copy(i_b_matrix[i_b_matrix[:, -1].argsort()][0, :])
        if (best_global[-1] > value[-1]):
            best_global = np.copy(value)
        if (decay > 0):
            n = decay
            w = w * (1 - ((count - 1)**n) / (iterations**n))
            c1 = (1 - c1) * (count / iterations) + c1
            c2 = (1 - c2) * (count / iterations) + c2
        init_velocity = velocity_vector(position,
                                        init_velocity,
                                        i_b_matrix,
                                        best_global,
                                        w=w,
                                        c1=c1,
                                        c2=c2)
        count = count + 1
        fitness_values.append({
            'fitness': best_global[-1],
            'accuracy': best_global[-2],
            'selected_features': best_global[-3],
            'selected_rate': best_global[-4]
        })
    return best_global, fitness_values


############################################################################
