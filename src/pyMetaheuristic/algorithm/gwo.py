# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Grey Wolf Optimizer

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

# Required Libraries
import random

import numpy as np


# Function
def target_function():
    return


# Function: Initialize Variables
def initial_position(pack_size=5,
                     min_values=[-5, -5],
                     max_values=[5, 5],
                     target_function=target_function,
                     target_function_parameters=None):
    position = np.zeros((pack_size, len(min_values) + 4))
    for i in range(0, pack_size):
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


def v_shaped_transfer_function(x, is_x_vector=False):
    if is_x_vector:
        threshold = np.random.random(x.shape)
        return np.where(hyperbolic_tan(x) > threshold, 1 - x, x)
    else:
        threshold = np.random.rand()
        return 1 - x if hyperbolic_tan(x) > threshold else x


def hyperbolic_tan(x):
    return np.abs(np.tanh(x))


# Function: Initialize Alpha
def alpha_position(min_values, max_values, target_function,
                   target_function_parameters):
    alpha = np.zeros((1, len(min_values) + 4))
    target_function_parameters['weights'] = np.clip(
        alpha[0, 0:alpha.shape[1] - 4], min_values, max_values)
    fitness = target_function(**target_function_parameters)
    alpha[0, -1] = fitness['fitness']
    alpha[0, -2] = fitness['accuracy']
    alpha[0, -3] = fitness['selected_features']
    alpha[0, -4] = fitness['selected_rate']

    return alpha[0, :]


# Function: Initialize Beta
def beta_position(min_values, max_values, target_function,
                  target_function_parameters):
    beta = np.zeros((1, len(min_values) + 4))
    target_function_parameters['weights'] = np.clip(
        beta[0, 0:beta.shape[1] - 4], min_values, max_values)
    fitness = target_function(**target_function_parameters)
    beta[0, -1] = fitness['fitness']
    beta[0, -2] = fitness['accuracy']
    beta[0, -3] = fitness['selected_features']
    beta[0, -4] = fitness['selected_rate']

    return beta[0, :]


# Function: Initialize Delta
def delta_position(min_values, max_values, target_function,
                   target_function_parameters):
    delta = np.zeros((1, len(min_values) + 4))
    target_function_parameters['weights'] = np.clip(
        delta[0, 0:delta.shape[1] - 4], min_values, max_values)
    fitness = target_function(**target_function_parameters)
    delta[0, -1] = fitness['fitness']
    delta[0, -2] = fitness['accuracy']
    delta[0, -3] = fitness['selected_features']
    delta[0, -4] = fitness['selected_rate']

    return delta[0, :]


# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    idx = np.argsort(position[:, -1])
    alpha = position[idx[0], :]
    beta = position[idx[1], :] if position.shape[0] > 1 else alpha
    delta = position[idx[2], :] if position.shape[0] > 2 else beta
    return alpha, beta, delta


# Function: Update Position
def update_position(position,
                    alpha,
                    beta,
                    delta,
                    a_linear_component=2,
                    min_values=[-5, -5],
                    max_values=[5, 5],
                    target_function=target_function,
                    target_function_parameters=None,
                    binary='r'):
    dim = len(min_values)
    alpha_position = np.copy(position)
    beta_position = np.copy(position)
    delta_position = np.copy(position)
    updated_position = np.copy(position)
    r1 = np.random.rand(position.shape[0], dim)
    r2 = np.random.rand(position.shape[0], dim)
    a = 2 * a_linear_component * r1 - a_linear_component
    c = 2 * r2
    distance_alpha = np.abs(c * alpha[:dim] - position[:, :dim])
    distance_beta = np.abs(c * beta[:dim] - position[:, :dim])
    distance_delta = np.abs(c * delta[:dim] - position[:, :dim])
    x1 = alpha[:dim] - a * distance_alpha
    x2 = beta[:dim] - a * distance_beta
    x3 = delta[:dim] - a * distance_delta

    alpha_position[:, :-4] = np.clip(x1, min_values, max_values)
    beta_position[:, :-4] = np.clip(x2, min_values, max_values)
    delta_position[:, :-4] = np.clip(x3, min_values, max_values)

    for i in range(alpha_position.shape[0]):
        target_function_parameters['weights'] = np.clip(
            alpha_position[i, :-4], min_values, max_values)
        fitness = target_function(**target_function_parameters)
        alpha_position[i, -1] = fitness['fitness']
        alpha_position[i, -2] = fitness['accuracy']
        alpha_position[i, -3] = fitness['selected_features']
        alpha_position[i, -4] = fitness['selected_rate']

        target_function_parameters['weights'] = np.clip(
            beta_position[i, :-4], min_values, max_values)
        fitness = target_function(**target_function_parameters)
        beta_position[i, -1] = fitness['fitness']
        beta_position[i, -2] = fitness['accuracy']
        beta_position[i, -3] = fitness['selected_features']
        beta_position[i, -4] = fitness['selected_rate']

        target_function_parameters['weights'] = np.clip(
            delta_position[i, :-4], min_values, max_values)
        fitness = target_function(**target_function_parameters)
        delta_position[i, -1] = fitness['fitness']
        delta_position[i, -2] = fitness['accuracy']
        delta_position[i, -3] = fitness['selected_features']
        delta_position[i, -4] = fitness['selected_rate']

    if binary == 's':
        updated_position[:, :-4] = s_shaped_transfer_function(
            (alpha_position[:, :-4] + beta_position[:, :-4] +
             delta_position[:, :-4]) / 3,
            is_x_vector=True)
    elif binary == 'v':
        updated_position[:, :-4] = v_shaped_transfer_function(
            (alpha_position[:, :-4] + beta_position[:, :-4] +
             delta_position[:, :-4]) / 3,
            is_x_vector=True)
    else:
        updated_position[:, :-4] = np.clip(
            (alpha_position[:, :-4] + beta_position[:, :-4] +
             delta_position[:, :-4]) / 3, min_values, max_values)

    for i in range(alpha_position.shape[0]):
        target_function_parameters['weights'] = np.clip(
            updated_position[i, :-4], min_values, max_values)
        fitness = target_function(**target_function_parameters)
        updated_position[i, -1] = fitness['fitness']
        updated_position[i, -2] = fitness['accuracy']
        updated_position[i, -3] = fitness['selected_features']
        updated_position[i, -4] = fitness['selected_rate']

    updated_position = np.vstack([
        position, updated_position, alpha_position, beta_position,
        delta_position
    ])
    updated_position = updated_position[updated_position[:, -1].argsort()]
    updated_position = updated_position[:position.shape[0], :]
    return updated_position


# GWO Function
def grey_wolf_optimizer(pack_size=5,
                        min_values=[-5, -5],
                        max_values=[5, 5],
                        iterations=50,
                        target_function=target_function,
                        verbose=True,
                        target_function_parameters=None,
                        binary='r'):
    alpha = alpha_position(min_values, max_values, target_function,
                           target_function_parameters)
    beta = beta_position(min_values, max_values, target_function,
                         target_function_parameters)
    delta = delta_position(min_values, max_values, target_function,
                           target_function_parameters)
    position = initial_position(pack_size, min_values, max_values,
                                target_function, target_function_parameters)
    count = 0
    fitness_values = []
    while (count <= iterations):
        if (verbose):
            print('Iteration = ', count, ' f(x) = ', alpha[-1])
        a_linear_component = 2 - count * (2 / iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position = update_position(position, alpha, beta, delta,
                                   a_linear_component, min_values, max_values,
                                   target_function, target_function_parameters,
                                   binary)
        fitness_values.append({
            'fitness': alpha[-1],
            'accuracy': alpha[-2],
            'selected_features': alpha[-3],
            'selected_rate': alpha[-4]
        })

        count += 1
    return alpha, fitness_values
