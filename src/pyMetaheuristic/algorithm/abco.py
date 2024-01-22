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
import os

############################################################################

# Function


def target_function():
    return

############################################################################

# Function: Initialize Variables


def initial_sources(food_sources=3, min_values=[-5, -5], max_values=[5, 5], target_function=target_function, target_function_parameters=None):
    sources = np.zeros((food_sources, len(min_values) + 2))
    for i in range(0, food_sources):
        for j in range(0, len(min_values)):
            sources[i, j] = random.uniform(min_values[j], max_values[j])
        target_function_parameters['weights'] = sources[i, :-2]
        fitness_values = target_function(**target_function_parameters)
        sources[i, -1] = fitness_values['ValFitness']
        sources[i, -2] = fitness_values['TrainFitness']
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
    return 1-x if hyperbolic_tan(x) > threshold else x


def hyperbolic_tan(x):
    return np.abs(np.tanh(x))

############################################################################

# Function: Fitness Value


def fitness_calc(function_value):
    if (function_value >= 0):
        fitness_value = 1.0/(1.0 + function_value)
    else:
        fitness_value = 1.0 + abs(function_value)
    return fitness_value

# Function: Fitness


def fitness_function(searching_in_sources):
    fitness = np.zeros((searching_in_sources.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i, 0] = fitness_calc(searching_in_sources[i, -1])
    fit_sum = fitness[:, 0].sum()
    fitness[0, 1] = fitness[0, 0]
    for i in range(1, fitness.shape[0]):
        fitness[i, 1] = (fitness[i, 0] + fitness[i-1, 1])
    for i in range(0, fitness.shape[0]):
        fitness[i, 1] = fitness[i, 1]/fit_sum
    return fitness

# Function: Selection


def roulette_wheel(fitness):
    ix = 0
    random = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
            ix = i
            break
    return ix

############################################################################

# Function: Employed Bee


def employed_bee(sources, min_values=[-5, -5], max_values=[5, 5], target_function=target_function, target_function_parameters=None):
    searching_in_sources = np.copy(sources)
    new_solution = np.zeros((1, len(min_values)+2))
    trial = np.zeros((sources.shape[0], 1))
    for i in range(0, searching_in_sources.shape[0]):
        phi = random.uniform(-1, 1)
        j = np.random.randint(len(min_values), size=1)[0]
        k = np.random.randint(searching_in_sources.shape[0], size=1)[0]
        while i == k:
            k = np.random.randint(searching_in_sources.shape[0], size=1)[0]
        xij = searching_in_sources[i, j]
        xkj = searching_in_sources[k, j]
        vij = xij + phi*(xij - xkj)
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = searching_in_sources[i, variable]
        new_solution[0, j] = np.clip(vij, min_values[j], max_values[j])
        target_function_parameters['weights'] = new_solution[0, :-2]
        fitness_values = target_function(**target_function_parameters)
        new_solution[0, -1] = fitness_values['ValFitness']
        new_solution[0, -2] = fitness_values['TrainFitness']
        new_function_value = new_solution[0, -1]
        if (fitness_calc(new_function_value) > fitness_calc(searching_in_sources[i, -1])):
            searching_in_sources[i, j] = new_solution[0, j]
            searching_in_sources[i, -1] = new_function_value
        else:
            trial[i, 0] = trial[i, 0] + 1
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = 0.0
    return searching_in_sources, trial

# Function: Oulooker


def outlooker_bee(searching_in_sources, fitness, trial, min_values=[-5, -5], max_values=[5, 5], target_function=target_function, target_function_parameters=None):
    improving_sources = np.copy(searching_in_sources)
    new_solution = np.zeros((1, len(min_values)+2))
    trial_update = np.copy(trial)
    for _ in range(0, improving_sources.shape[0]):
        i = roulette_wheel(fitness)
        phi = random.uniform(-1, 1)
        j = np.random.randint(len(min_values), size=1)[0]
        k = np.random.randint(improving_sources.shape[0], size=1)[0]
        while i == k:
            k = np.random.randint(improving_sources.shape[0], size=1)[0]
        xij = improving_sources[i, j]
        xkj = improving_sources[k, j]
        vij = xij + phi*(xij - xkj)
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = improving_sources[i, variable]
        new_solution[0, j] = np.clip(vij,  min_values[j], max_values[j])
        target_function_parameters['weights'] = new_solution[0, :-2]
        fitness_values = target_function(**target_function_parameters)
        new_solution[0, -1] = fitness_values['ValFitness']
        new_solution[0, -2] = fitness_values['TrainFitness']
        new_function_value = new_solution[0, -1]
        if (fitness_calc(new_function_value) > fitness_calc(improving_sources[i, -1])):
            improving_sources[i, j] = new_solution[0, j]
            improving_sources[i, -1] = new_function_value
            trial_update[i, 0] = 0
        else:
            trial_update[i, 0] = trial_update[i, 0] + 1
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = 0.0
    return improving_sources, trial_update

# Function: Scouter


def scouter_bee(improving_sources, trial_update, limit=3, target_function=target_function, target_function_parameters=None, binary='s'):
    for i in range(0, improving_sources.shape[0]):
        if (trial_update[i, 0] > limit):
            for j in range(0, improving_sources.shape[1] - 2):
                if binary == 's':
                    improving_sources[i, j] = s_shaped_transfer_function(
                        np.random.normal(0, 1, 1)[0])
                elif binary == 'v':
                    improving_sources[i, j] = v_shaped_transfer_function(
                        np.random.normal(0, 1, 1)[0])
                else:
                    improving_sources[i, j] = np.random.normal(0, 1, 1)[0]
            target_function_parameters['weights'] = improving_sources[i, :-2]
            fitness_values = target_function(**target_function_parameters)
            improving_sources[i, -1] = fitness_values['ValFitness']
            improving_sources[i, -2] = fitness_values['TrainFitness']
            function_value = improving_sources[i, -1]
            improving_sources[i, -1] = function_value
    return improving_sources

############################################################################

# ABCO Function


def artificial_bee_colony_optimization(food_sources=3, iterations=50, min_values=[-5, -5], max_values=[5, 5], employed_bees=3, outlookers_bees=3, limit=3, target_function=target_function, verbose=True, target_function_parameters=None, binary='s'):
    count = 0
    best_value = float('inf')
    sources = initial_sources(food_sources, min_values,
                              max_values, target_function, target_function_parameters)
    fitness_values = []
    fitness = fitness_function(sources)
    while (count <= iterations):
        if (count > 0):
            if (verbose == True):
                print('Iteration = ', count, ' f(x) = ', best_value)
        e_bee = employed_bee(sources, min_values, max_values,
                             target_function, target_function_parameters)
        for _ in range(0, employed_bees - 1):
            e_bee = employed_bee(
                e_bee[0], min_values, max_values, target_function, target_function_parameters)
        fitness = fitness_function(e_bee[0])
        o_bee = outlooker_bee(
            e_bee[0], fitness, e_bee[1], min_values, max_values, target_function, target_function_parameters)
        for _ in range(0, outlookers_bees - 1):
            o_bee = outlooker_bee(
                o_bee[0], fitness, o_bee[1], min_values, max_values, target_function, target_function_parameters)
        value = np.copy(o_bee[0][o_bee[0][:, -1].argsort()][0, :])
        if (best_value > value[-1]):
            best_solution = np.copy(value)
            best_value = np.copy(value[-1])
        sources = scouter_bee(
            o_bee[0], o_bee[1], limit=limit, target_function=target_function, target_function_parameters=target_function_parameters, binary=binary)
        fitness = fitness_function(sources)
        fitness_values.append(
            {'ValFitness': best_solution[-1], 'TrainFitness': best_solution[-2]})
        count = count + 1

        print(best_solution)
    return best_solution, fitness_values

############################################################################
