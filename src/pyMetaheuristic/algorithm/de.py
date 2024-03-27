############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Differential Evolution

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import random

import numpy as np

############################################################################


# Function
def target_function():
    return


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
def initial_position(n=3,
                     min_values=[-5, -5],
                     max_values=[5, 5],
                     target_function=target_function,
                     target_function_parameters=None):
    position = np.zeros((n, len(min_values) + 4))
    for i in range(0, n):
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


# Function: Velocity
def velocity(position, best_global, k0, k1, k2, F, min_values, max_values, Cr,
             target_function, target_function_parameters, binary):
    v = np.copy(best_global)
    for i in range(0, len(best_global)):
        ri = np.random.rand()
        if (ri <= Cr):
            v[i] = best_global[i] + F * (position[k1, i] - position[k2, i])
        else:
            if binary == 's':
                v[i] = s_shaped_transfer_function(position[k0, i])
            elif binary == 'v':
                v[i] = v_shaped_transfer_function(position[k0, i])
            else:
                v[i] = position[k0, i]
        if (i < len(min_values) and v[i] > max_values[i]):
            v[i] = max_values[i]
        elif (i < len(min_values) and v[i] < min_values[i]):
            v[i] = min_values[i]
    target_function_parameters['weights'] = v[0:len(v) - 4]
    fitness = target_function(**target_function_parameters)
    v[-1] = fitness['fitness']
    v[-2] = fitness['accuracy']
    v[-3] = fitness['selected_features']
    v[-4] = fitness['selected_rate']
    return v


############################################################################


# Function: DE/Best/1/Bin Scheme
def differential_evolution(n=3,
                           min_values=[-5, -5],
                           max_values=[5, 5],
                           iterations=50,
                           F=0.9,
                           Cr=0.2,
                           target_function=target_function,
                           target_function_parameters=None,
                           binary='s',
                           verbose=True):
    position = initial_position(n, min_values, max_values, target_function,
                                target_function_parameters)
    best_global = np.copy(position[position[:, -1].argsort()][0, :])
    count = 0
    metrics = []
    while (count <= iterations):
        if (verbose):
            print('Iteration = ', count, ' f(x) ', best_global[-1])
        for i in range(0, position.shape[0]):
            k1 = int(np.random.randint(position.shape[0], size=1))
            k2 = int(np.random.randint(position.shape[0], size=1))
            while k1 == k2:
                k1 = int(np.random.randint(position.shape[0], size=1))
            vi = velocity(position, best_global, i, k1, k2, F, min_values,
                          max_values, Cr, target_function,
                          target_function_parameters, binary)
            if (vi[-1] <= position[i, -1]):
                for j in range(0, position.shape[1]):
                    position[i, j] = vi[j]
            if (best_global[-1] > position[position[:,
                                                    -1].argsort()][0, :][-1]):
                best_global = np.copy(position[position[:,
                                                        -1].argsort()][0, :])

        count += 1
        metrics.append({
            'fitness': best_global[-1],
            'accuracy': best_global[-2],
            'selected_features': best_global[-3],
            'selected_rate': best_global[-4]
        })
    return best_global, metrics


############################################################################
