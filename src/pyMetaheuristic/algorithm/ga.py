############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Genetic Algorithm

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


# Function: Initialize Variables
def initial_population(population_size=5,
                       min_values=[-5, -5],
                       max_values=[5, 5],
                       target_function=target_function,
                       target_function_parameters=None,
                       binary=True):
    population = np.zeros((population_size, len(min_values) + 4))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
            if binary:
                population[i, j] = random.choice([0, 1])
            else:
                population[i, j] = random.uniform(min_values[j], max_values[j])
        target_function_parameters['weights'] = population[
            i, 0:population.shape[1] - 4]
        fitness = target_function(**target_function_parameters)
        population[i, -1] = fitness['fitness']
        population[i, -2] = fitness['accuracy']
        population[i, -3] = fitness['selected_features']
        population[i, -4] = fitness['selected_rate']
    return population


############################################################################


def fitness_function(population):
    min_pop = abs(population[:, -1].min())
    fitness_first_col = 1 / (1 + population[:, -1] + min_pop)
    fitness_second_col = np.cumsum(fitness_first_col)
    fitness_second_col = fitness_second_col / fitness_second_col[-1]
    fitness = np.column_stack((fitness_first_col, fitness_second_col))
    return fitness


# Function: Selection
def roulette_wheel(fitness):
    ix = 0
    random = np.random.rand()
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
            ix = i
            break
    return ix


def blend_crossover(parent_1, parent_2, alpha=0.5):
    child1 = parent_1[:]
    child2 = parent_2[:]

    for i in range(2):
        min_val = min(parent_1[i], parent_2[i])
        max_val = max(parent_1[i], parent_2[i])
        range_ = max_val - min_val
        low = min_val - alpha * range_
        high = max_val + alpha * range_

        # Create offspring
        child1[:-4] = np.random.uniform(low, high)
        child2[:-4] = np.random.uniform(low, high)

    return child1, child2


def one_point_crossover(parent_1, parent_2):
    # Randomly choose the crossover point
    crossover_point = np.random.randint(1, len(parent_1) - 1)

    # Create offspring using one-point crossover
    child1 = np.concatenate(
        (parent_1[:crossover_point], parent_2[crossover_point:]))
    child2 = np.concatenate(
        (parent_2[:crossover_point], parent_1[crossover_point:]))

    return child1, child2


# Function: Breeding
def breeding(population,
             fitness,
             crossover_rate,
             elite,
             alpha,
             target_function,
             target_function_parameters=None,
             binary=True):
    offspring = np.copy(population)

    for _ in range(elite, offspring.shape[0], 2):
        # Get two worst individual
        sorted_offspring_indices = np.argsort(offspring[:, -1])
        worst_idx0, worst_idx1 = sorted_offspring_indices[
            -1], sorted_offspring_indices[-2]
        parent_1_idx, parent_2_idx = roulette_wheel(fitness), roulette_wheel(
            fitness)

        # Ensure parents are different
        while parent_1_idx == parent_2_idx:
            parent_2_idx = np.random.choice(len(population) - 1)

        parent_1 = population[parent_1_idx, :]
        parent_2 = population[parent_2_idx, :]

        # Check if crossover should occur
        if np.random.rand() < crossover_rate:
            if binary:
                child1, child2 = one_point_crossover(parent_1, parent_2)
            else:
                child1, child2 = blend_crossover(parent_1, parent_2, alpha)

            # Evaluate fitness for the offspring
            target_function_parameters['weights'] = child1[:-4]
            fitness_values_child1 = target_function(
                **target_function_parameters)
            child1[-1] = fitness_values_child1['fitness']
            child1[-2] = fitness_values_child1['accuracy']
            child1[-3] = fitness_values_child1['selected_features']
            child1[-4] = fitness_values_child1['selected_rate']

            target_function_parameters['weights'] = child2[:-4]
            fitness_values_child2 = target_function(
                **target_function_parameters)
            child2[-1] = fitness_values_child2['fitness']
            child2[-2] = fitness_values_child2['accuracy']
            child2[-3] = fitness_values_child2['selected_features']
            child2[-4] = fitness_values_child2['selected_rate']

            # Check if the child's fitness is better than the worst individuals
            if child1[-1] < offspring[worst_idx0, -1]:
                offspring[worst_idx0] = child1[:]

            if child2[-1] < offspring[worst_idx1, -1]:
                offspring[worst_idx1] = child2[:]

        else:
            # If no crossover, offspring are identical to parents
            offspring[parent_1_idx] = parent_1[:]
            offspring[parent_2_idx] = parent_2[:]

        if elite > 0:
            offspring = offspring[offspring[:, -1].argsort()]
            preserve = np.copy(offspring[offspring[:, -1].argsort()])
            offspring[
                -elite:, :] = preserve[:
                                       elite, :]  # The best individuals are keep and the worst are replaced

    return offspring


# Function: Mutation
def mutation(offspring,
             eta=1,
             min_values=[-5, -5],
             max_values=[5, 5],
             mutation_rate=0.1,
             target_function=target_function,
             target_function_parameters=None,
             binary=True):
    for i in range(0, offspring.shape[0]):
        p = random.uniform(0, 1)
        random_idx = random.randint(0, offspring.shape[1] - 3)

        if binary:
            bit = offspring[i, random_idx]
            offspring[i, random_idx] = bit if p < mutation_rate else 1 - bit
        else:
            rand = np.random.rand()
            rand_d = np.random.rand()
            if rand <= 0.5:
                d_mutation = 2 * (rand_d)
                d_mutation = d_mutation**(1 / (eta + 1)) - 1
            elif rand > 0.5:
                d_mutation = 2 * (1 - rand_d)
                d_mutation = 1 - d_mutation**(1 / (eta + 1))
            offspring[i, random_idx] = np.clip(
                (offspring[i, random_idx] + d_mutation),
                min_values[random_idx], max_values[random_idx])

        target_function_parameters['weights'] = offspring[i, 0:-4]
        fitness_values = target_function(**target_function_parameters)
        offspring[i, -1] = fitness_values['fitness']
        offspring[i, -2] = fitness_values['accuracy']
        offspring[i, -3] = fitness_values['selected_features']
        offspring[i, -4] = fitness_values['selected_rate']

    return offspring


############################################################################


# GA Function
def genetic_algorithm(population_size=5,
                      mutation_rate=0.1,
                      elite=0,
                      alpha=0.5,
                      min_values=[-5, -5],
                      max_values=[5, 5],
                      crossover_rate=0.8,
                      generations=50,
                      eta=1,
                      target_function=target_function,
                      target_function_parameters=None,
                      verbose=True,
                      binary=True):
    count = 0
    fitness_values = []

    population = initial_population(population_size, min_values, max_values,
                                    target_function,
                                    target_function_parameters, binary)
    fitness = fitness_function(population)
    elite_ind = np.copy(population[population[:, -1].argsort()][0, :])
    while (count <= generations):
        if (verbose):
            print('Generation = ', count, ' f(x) = ', elite_ind[-1])
        offspring = breeding(population, fitness, crossover_rate, elite, alpha,
                             target_function, target_function_parameters,
                             binary)
        population = mutation(offspring, eta, min_values, max_values,
                              mutation_rate, target_function,
                              target_function_parameters, binary)
        fitness = fitness_function(population)
        value = np.copy(population[population[:, -1].argsort()][0, :])
        if (elite_ind[-1] > value[-1]):
            elite_ind = np.copy(value)
        count = count + 1
        fitness_values.append({
            'fitness': elite_ind[-1],
            'accuracy': elite_ind[-2],
            'selected_features': elite_ind[-3],
            'selected_rate': elite_ind[-4]
        })

    return elite_ind, fitness_values


############################################################################
