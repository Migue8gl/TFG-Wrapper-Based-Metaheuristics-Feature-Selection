############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Grey Wolf Optimizer

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

# Function: Initialize Variables


def initial_position(pack_size=5, min_values=[-5, -5], max_values=[5, 5], target_function=target_function, target_function_parameters=None):
    position = np.zeros((pack_size, len(min_values)+2))
    for i in range(0, pack_size):
        for j in range(0, len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        target_function_parameters['weights'] = position[i,
                                                         0:position.shape[1]-2]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['ValFitness']
        position[i, -2] = fitness['TrainFitness']
    return position

############################################################################

# Transfer functions S-Shaped


def s_shaped_transfer_function(x):
    threshold = np.random.rand()
    return 1 if sigmoid(x) > threshold else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-10*(x-0.5)))

############################################################################

# Transfer functions V-Shaped


def v_shaped_transfer_function(x):
    threshold = np.random.rand()
    return 1-x if hyperbolic_tan(x) > threshold else x


def hyperbolic_tan(x):
    return np.abs(np.tanh(x))

############################################################################

# Function: Initialize Alpha


def alpha_position(dimension=2, target_function=target_function, target_function_parameters=None):
    alpha = np.zeros((1, dimension + 2))
    for j in range(0, dimension):
        alpha[0, j] = 0.0
    target_function_parameters['weights'] = alpha[0, 0:alpha.shape[1]-2]
    fitness = target_function(**target_function_parameters)
    alpha[0, -1] = fitness['ValFitness']
    alpha[0, -2] = fitness['TrainFitness']
    return alpha

# Function: Initialize Beta


def beta_position(dimension=2, target_function=target_function, target_function_parameters=None):
    beta = np.zeros((1, dimension + 2))
    for j in range(0, dimension):
        beta[0, j] = 0.0
    target_function_parameters['weights'] = beta[0, 0:beta.shape[1]-2]
    fitness = target_function(**target_function_parameters)
    beta[0, -1] = fitness['ValFitness']
    beta[0, -2] = fitness['TrainFitness']
    return beta

# Function: Initialize Delta


def delta_position(dimension=2, target_function=target_function, target_function_parameters=None):
    delta = np.zeros((1, dimension + 2))
    for j in range(0, dimension):
        delta[0, j] = 0.0
    target_function_parameters['weights'] = delta[0, 0:delta.shape[1]-2]
    fitness = target_function(**target_function_parameters)
    delta[0, -1] = fitness['ValFitness']
    delta[0, -2] = fitness['TrainFitness']
    return delta

# Function: Updtade Pack by Fitness


def update_pack(position, alpha, beta, delta):
    updated_position = np.copy(position)
    for i in range(0, position.shape[0]):
        if (updated_position[i, -1] < alpha[0, -1]):
            alpha[0, :] = np.copy(updated_position[i, :])
        if (updated_position[i, -1] > alpha[0, -1] and updated_position[i, -1] < beta[0, -1]):
            beta[0, :] = np.copy(updated_position[i, :])
        if (updated_position[i, -1] > alpha[0, -1] and updated_position[i, -1] > beta[0, -1] and updated_position[i, -1] < delta[0, -1]):
            delta[0, :] = np.copy(updated_position[i, :])
    return alpha, beta, delta

# Function: Updtade Position


def update_position(position, alpha, beta, delta, a_linear_component=2, min_values=[-5, -5], max_values=[5, 5], target_function=target_function, target_function_parameters=None, binary='x'):
    updated_position = np.copy(position)
    for i in range(0, updated_position.shape[0]):
        for j in range(0, len(min_values)):
            r1_alpha = int.from_bytes(os.urandom(
                8), byteorder='big') / ((1 << 64) - 1)
            r2_alpha = int.from_bytes(os.urandom(
                8), byteorder='big') / ((1 << 64) - 1)
            a_alpha = 2*a_linear_component*r1_alpha - a_linear_component
            c_alpha = 2*r2_alpha
            distance_alpha = abs(c_alpha*alpha[0, j] - position[i, j])
            x1 = alpha[0, j] - a_alpha*distance_alpha
            r1_beta = int.from_bytes(os.urandom(
                8), byteorder='big') / ((1 << 64) - 1)
            r2_beta = int.from_bytes(os.urandom(
                8), byteorder='big') / ((1 << 64) - 1)
            a_beta = 2*a_linear_component*r1_beta - a_linear_component
            c_beta = 2*r2_beta
            distance_beta = abs(c_beta*beta[0, j] - position[i, j])
            x2 = beta[0, j] - a_beta*distance_beta
            r1_delta = int.from_bytes(os.urandom(
                8), byteorder='big') / ((1 << 64) - 1)
            r2_delta = int.from_bytes(os.urandom(
                8), byteorder='big') / ((1 << 64) - 1)
            a_delta = 2*a_linear_component*r1_delta - a_linear_component
            c_delta = 2*r2_delta
            distance_delta = abs(c_delta*delta[0, j] - position[i, j])
            x3 = delta[0, j] - a_delta*distance_delta
            if binary == 's':
                updated_position[i, j] = s_shaped_transfer_function(
                    np.clip((x1 + x2 + x3)/3, min_values[j], max_values[j]))
            elif binary == 'v':
                updated_position[i, j] = v_shaped_transfer_function(
                    np.clip((x1 + x2 + x3)/3, min_values[j], max_values[j]))
            else:
                updated_position[i, j] = np.clip(
                    ((x1 + x2 + x3)/3), min_values[j], max_values[j])
        target_function_parameters['weights'] = updated_position[i,
                                                                 0:updated_position.shape[1]-2]
        fitness = target_function(**target_function_parameters)
        updated_position[i, -1] = fitness['ValFitness']
        updated_position[i, -2] = fitness['TrainFitness']

    return updated_position

############################################################################

# GWO Function


def grey_wolf_optimizer(pack_size=5, min_values=[-5, -5], max_values=[5, 5], iterations=50, target_function=target_function, verbose=True, target_function_parameters=None, binary='x'):
    count = 0
    alpha = alpha_position(dimension=len(min_values), target_function=target_function,
                           target_function_parameters=target_function_parameters)
    beta = beta_position(dimension=len(min_values), target_function=target_function,
                         target_function_parameters=target_function_parameters)
    delta = delta_position(dimension=len(min_values), target_function=target_function,
                           target_function_parameters=target_function_parameters)
    position = initial_position(pack_size=pack_size, min_values=min_values, max_values=max_values,
                                target_function=target_function, target_function_parameters=target_function_parameters)
    fitness_values = []
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', alpha[0][-1])
        a_linear_component = 2 - count*(2/iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position = update_position(position, alpha, beta, delta, a_linear_component=a_linear_component, min_values=min_values,
                                   max_values=max_values, target_function=target_function, target_function_parameters=target_function_parameters, binary=binary)
        count = count + 1
        fitness_values.append(
            {'ValFitness': alpha[0, -1], 'TrainFitness': alpha[0, -2]})
    return alpha.flatten(), fitness_values

############################################################################
