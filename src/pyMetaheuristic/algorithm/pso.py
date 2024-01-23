############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Particle Swarm Optimization

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np
import random
import os

############################################################################

# Function


def target_function():
    return


############################################################################

# Transfer functions S-Shaped


def s_shaped_transfer_function(x):
    threshold = np.random.rand()
    return 1 if sigmoid(x) > threshold else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Transfer functions S-Shaped


def v_shaped_transfer_function(x, delta_x):
    threshold = np.random.rand()
    return 1 - x if hyperbolic(delta_x) > threshold else x


def hyperbolic(x):
    return np.abs(x / np.sqrt(x**2 + 1))


############################################################################

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
        target_function_parameters['weights'] = position[i, :-2]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['ValFitness']
        position[i, -2] = fitness['TrainFitness']
    return position


############################################################################

# Function: Initialize Velocity


def initial_velocity(position, min_values=[-5, -5], max_values=[5, 5]):
    init_velocity = np.zeros((position.shape[0], len(min_values)))
    for i in range(0, init_velocity.shape[0]):
        for j in range(0, init_velocity.shape[1]):
            init_velocity[i, j] = random.uniform(min_values[j], max_values[j])
    return init_velocity


# Function: Individual Best


def individual_best_matrix(position, i_b_matrix):
    for i in range(0, position.shape[0]):
        if (i_b_matrix[i, -1] > position[i, -1]):
            for j in range(0, position.shape[1]):
                i_b_matrix[i, j] = position[i, j]
    return i_b_matrix


# Function: Velocity


def velocity_vector(position,
                    init_velocity,
                    i_b_matrix,
                    best_global,
                    w=0.5,
                    c1=2,
                    c2=2):
    r1 = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
    r2 = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
    velocity = np.zeros((position.shape[0], init_velocity.shape[1]))
    for i in range(0, init_velocity.shape[0]):
        for j in range(0, init_velocity.shape[1]):
            velocity[i, j] = w*init_velocity[i, j] + c1*r1 * \
                (i_b_matrix[i, j] - position[i, j]) + \
                c2*r2*(best_global[j] - position[i, j])
    return velocity


# Function: Updtade Position


def update_position(position,
                    velocity,
                    min_values=[-5, -5],
                    max_values=[5, 5],
                    target_function=target_function,
                    target_function_parameters=None,
                    binary='s'):
    for i in range(0, position.shape[0]):
        for j in range(0, position.shape[1] - 2):
            if binary == 's':
                position[i, j] = s_shaped_transfer_function(velocity[i, j])
            elif binary == 'v':
                position[i, j] = v_shaped_transfer_function(
                    position[i, j], velocity[i, j])
            else:
                position[i, j] = np.clip((position[i, j] + velocity[i, j]),
                                         min_values[j], max_values[j])
        target_function_parameters['weights'] = position[i, :-2]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['ValFitness']
        position[i, -2] = fitness['TrainFitness']
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
        if (verbose == True):
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
            'ValFitness': best_global[-1],
            'TrainFitness': best_global[-2]
        })
    return best_global, fitness_values


############################################################################
