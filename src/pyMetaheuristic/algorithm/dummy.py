import random

import numpy as np


def target_function():
    pass


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


def dummy_optimizer(swarm_size=3,
                    min_values=[-5, -5],
                    max_values=[5, 5],
                    iterations=50,
                    target_function=target_function,
                    target_function_parameters=None,
                    binary='s',
                    verbose=True):
    count = 0
    fitness_values = []
    position = np.zeros((swarm_size, len(min_values) + 4))
    for i in range(swarm_size):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        target_function_parameters['weights'] = position[i, :-4]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['fitness']
        position[i, -2] = fitness['accuracy']
        position[i, -3] = fitness['selected_features']
        position[i, -4] = fitness['selected_rate']

    best_global = np.copy(position[position[:, -1].argsort()][0, :])

    for _ in range(iterations):
        random_particle = random.randint(0, swarm_size - 1)
        random_dimension = random.randint(0, len(min_values) - 4)
        random_value = random.uniform(min_values[random_dimension],
                                      max_values[random_dimension])

        if (verbose):
            print('Iteration = ', count, ' f(x) = ', best_global[-1])

        if binary == 's':
            position[random_particle,
                     random_dimension] = s_shaped_transfer_function(
                         random_value)
        elif binary == 'v':
            position[random_particle,
                     random_dimension] = v_shaped_transfer_function(
                         random_value, position[random_particle,
                                                random_dimension])
        else:
            position[random_particle, random_dimension] = random_value

        target_function_parameters['weights'] = position[random_particle, :-4]
        fitness = target_function(**target_function_parameters)
        position[random_particle, -1] = fitness['fitness']
        position[random_particle, -2] = fitness['accuracy']
        position[random_particle, -3] = fitness['selected_features']
        position[random_particle, -4] = fitness['selected_rate']

        if position[random_particle, -1] < best_global[-1]:
            best_global = np.copy(position[random_particle])
        count += 1
        fitness_values.append({
            'fitness': best_global[-1],
            'accuracy': best_global[-2],
            'selected_features': best_global[-3],
            'selected_rate': best_global[-4]
        })

    return best_global, fitness_values
