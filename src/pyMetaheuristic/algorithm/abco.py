############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Artificial Bee Colony Optimization

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np
import random

############################################################################

# Function


def target_function():
    return


############################################################################

# Function: Initialize Variables


def initial_sources(food_sources=3,
                    min_values=[-5, -5],
                    max_values=[5, 5],
                    target_function=target_function,
                    target_function_parameters=None):
    sources = np.zeros((food_sources, len(min_values) + 2))
    for i in range(0, food_sources):
        for j in range(0, len(min_values)):
            sources[i, j] = random.uniform(min_values[j], max_values[j])
        target_function_parameters['weights'] = sources[i, :-2]
        fitness_values = target_function(**target_function_parameters)
        sources[i, -1] = fitness_values['validation']['fitness']
        sources[i, -2] = fitness_values['training']['fitness']
    return sources


############################################################################

# Transfer functions S-Shaped


def s_shaped_transfer_function(x):
    threshold = np.random.rand()
    return 1 if sigmoid(x) > threshold else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


############################################################################

# Transfer functions V-Shaped


def v_shaped_transfer_function(x):
    threshold = np.random.rand()
    return 1 - x if hyperbolic_tan(x) > threshold else x


def hyperbolic_tan(x):
    return np.abs(np.tanh(x))


############################################################################

# Function: Fitness Value


# Function: Fitness Value
def fitness_calc(function_values):
    fitness_value = np.where(function_values >= 0,
                             1.0 / (1.0 + function_values),
                             1.0 + np.abs(function_values))
    return fitness_value


# Function: Fitness
def fitness_function(sources, fitness_calc):
    fitness_values = fitness_calc(sources[:, -1])
    cumulative_sum = np.cumsum(fitness_values)
    normalized_cum_sum = cumulative_sum / cumulative_sum[-1]
    fitness = np.column_stack((fitness_values, normalized_cum_sum))
    return fitness


# Function: Selection
def roulette_wheel(fitness):
    ix = np.searchsorted(fitness[:, 1], np.random.rand())
    return ix


############################################################################

# Function: Employed Bee


def employed_bee(
    sources,
    min_values,
    max_values,
    target_function,
    target_function_parameters,
):
    searching_in_sources = np.copy(sources)
    dim = len(min_values) + 2
    trial = np.zeros((sources.shape[0], 1))
    new_solution = np.zeros((1, dim))
    phi_values = np.random.uniform(-1, 1, size=sources.shape[0])
    j_values = np.random.randint(dim-2, size=sources.shape[0])
    k_values = np.array([
        np.random.choice([k for k in range(sources.shape[0]) if k != i])
        for i in range(sources.shape[0])
    ])
    for i in range(0, sources.shape[0]):
        phi = phi_values[i]
        j = j_values[i]
        k = k_values[i]
        xij = searching_in_sources[i, j]
        xkj = searching_in_sources[k, j]
        vij = xij + phi * (xij - xkj)
        new_solution[0, :-2] = searching_in_sources[i, :-2]
        new_solution[0, j] = np.clip(vij, min_values[j], max_values[j])
        target_function_parameters['weights'] = new_solution[0, :-2]
        fitness_values = target_function(**target_function_parameters)
        new_solution[0, -1] = fitness_values['validation']['fitness']
        new_solution[0, -2] = fitness_values['training']['fitness']
        new_function_value = new_solution[0, -1]
        if (fitness_calc(new_function_value)
                > fitness_calc(searching_in_sources[i, -1])):
            searching_in_sources[i, j] = new_solution[0, j]
            searching_in_sources[i, -1] = new_function_value
        else:
            trial[i, 0] = trial[i, 0] + 1
    return searching_in_sources, trial


# Function: Oulooker


def outlooker_bee(searching_in_sources,
                  fitness,
                  trial,
                  min_values=[-5, -5],
                  max_values=[5, 5],
                  target_function=target_function,
                  target_function_parameters=None):
    improving_sources = np.copy(searching_in_sources)
    dim = len(min_values) + 2
    trial_update = np.copy(trial)
    new_solution = np.zeros((1, dim))
    phi_values = np.random.uniform(-1, 1, size=improving_sources.shape[0])
    j_values = np.random.randint(dim-2, size=improving_sources.shape[0])
    for repeat in range(0, improving_sources.shape[0]):
        i = roulette_wheel(fitness)
        phi = phi_values[repeat]
        j = j_values[repeat]
        k = np.random.choice(
            [k for k in range(0, improving_sources.shape[0]) if k != i])
        xij = improving_sources[i, j]
        xkj = improving_sources[k, j]
        vij = xij + phi * (xij - xkj)
        new_solution[0, :-2] = improving_sources[i, :-2]
        new_solution[0, j] = np.clip(vij, min_values[j], max_values[j])
        target_function_parameters['weights'] = new_solution[0, :-2]
        fitness_values = target_function(**target_function_parameters)
        new_solution[0, -1] = fitness_values['validation']['fitness']
        new_solution[0, -2] = fitness_values['training']['fitness']
        new_function_value = new_solution[0, -1]
        if (fitness_calc(new_function_value)
                > fitness_calc(improving_sources[i, -1])):
            improving_sources[i, j] = new_solution[0, j]
            improving_sources[i, -1] = new_function_value
            trial_update[i, 0] = 0
        else:
            trial_update[i, 0] = trial_update[i, 0] + 1
    return improving_sources, trial_update


# Function: Scouter


def scouter_bee(improving_sources,
                trial_update,
                limit=3,
                target_function=target_function,
                target_function_parameters=None,
                binary='s'):
    for i in range(0, improving_sources.shape[0]):
        if (trial_update[i, 0] > limit):
            for j in range(0, improving_sources.shape[1] - 2):
                if binary == 's':
                    improving_sources[i, j] = s_shaped_transfer_function(
                        np.random.normal(0, 1,
                                         improving_sources.shape[1] - 1)[0])
                elif binary == 'v':
                    improving_sources[i, j] = v_shaped_transfer_function(
                        np.random.normal(0, 1,
                                         improving_sources.shape[1] - 1)[0])
                else:
                    improving_sources[i, j] = np.random.normal(
                        0, 1, improving_sources.shape[1] - 1)[0]
            target_function_parameters['weights'] = improving_sources[i, :-2]
            fitness_values = target_function(**target_function_parameters)
            improving_sources[i, -1] = fitness_values['validation']['fitness']
            improving_sources[i, -2] = fitness_values['training']['fitness']
    return improving_sources


############################################################################

# ABCO Function


def artificial_bee_colony_optimization(food_sources=3,
                                       iterations=50,
                                       min_values=[-5, -5],
                                       max_values=[5, 5],
                                       employed_bees=3,
                                       outlookers_bees=3,
                                       limit=3,
                                       target_function=target_function,
                                       verbose=True,
                                       target_function_parameters=None,
                                       binary='s'):
    count = 0
    best_value = float('inf')
    sources = initial_sources(food_sources, min_values, max_values,
                              target_function, target_function_parameters)
    fitness_values = []
    fitness = fitness_function(sources, fitness_calc)
    while (count <= iterations):
        if (count > 0):
            if (verbose):
                print('Iteration = ', count, ' f(x) = ', best_value)
        e_bee = employed_bee(sources, min_values, max_values, target_function,
                             target_function_parameters)
        for _ in range(0, employed_bees - 1):
            e_bee = employed_bee(e_bee[0], min_values, max_values,
                                 target_function, target_function_parameters)
        fitness = fitness_function(e_bee[0], fitness_calc)
        o_bee = outlooker_bee(e_bee[0], fitness, e_bee[1], min_values,
                              max_values, target_function,
                              target_function_parameters)
        for _ in range(0, outlookers_bees - 1):
            o_bee = outlooker_bee(o_bee[0], fitness, o_bee[1], min_values,
                                  max_values, target_function,
                                  target_function_parameters)
        value = np.copy(o_bee[0][o_bee[0][:, -1].argsort()][0, :])
        if (best_value > value[-1]):
            best_solution = np.copy(value)
            best_value = np.copy(value[-1])
        sources = scouter_bee(
            o_bee[0],
            o_bee[1],
            limit=limit,
            target_function=target_function,
            target_function_parameters=target_function_parameters,
            binary=binary)
        fitness = fitness_function(sources, fitness_calc)
        fitness_values.append({
            'val_fitness': best_solution[-1],
            'train_fitness': best_solution[-2]
        })
        count = count + 1

    return best_solution, fitness_values


############################################################################
