############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Genetic Algorithm

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
def initial_population(population_size=5,
                       min_values=[-5, -5],
                       max_values=[5, 5],
                       target_function=target_function,
                       target_function_parameters=None):
    population = np.zeros((population_size, len(min_values) + 2))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
            population[i, j] = random.choice([0, 1])
        target_function_parameters['weights'] = population[
            i, 0:population.shape[1] - 2]
        fitness = target_function(**target_function_parameters)
        population[i, -1] = fitness['ValFitness']
        population[i, -2] = fitness['TrainFitness']
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


# Function: Breeding
def breeding(population,
             fitness,
             crossover_rate,
             elite,
             target_function,
             target_function_parameters=None):
    offspring = np.copy(population)
    offspring = offspring[np.argsort(offspring[:, -1])[::-1]]

    if elite > 0:
        preserve = np.copy(population[population[:, -1].argsort()])
        offspring[:elite, :] = preserve[:elite, :]

    for _ in range(elite, offspring.shape[0], 2):
        # Sort offspring only once before the loop
        sorted_offspring_indices = np.argsort(offspring[:, -1])[::-1]
        worst_individuals = sorted_offspring_indices[-2:]
        parent_1_idx, parent_2_idx = roulette_wheel(fitness), roulette_wheel(fitness)

        # Ensure parents are different
        while parent_1_idx == parent_2_idx:
            parent_2_idx = np.random.choice(len(population) - 1)

        parent_1 = population[parent_1_idx, :]
        parent_2 = population[parent_2_idx, :]

        # Check if crossover should occur
        if np.random.rand() < crossover_rate:
            # Randomly choose the crossover point
            crossover_point = np.random.randint(1, len(parent_1) - 1)

            # Create offspring using binary crossover
            child1 = np.copy(parent_1)
            child2 = np.copy(parent_2)

            child1[crossover_point:] = parent_2[crossover_point:]
            child2[crossover_point:] = parent_1[crossover_point:]

            # Evaluate fitness for the offspring
            target_function_parameters['weights'] = child1[:-2]
            fitness_values_child1 = target_function(**target_function_parameters)
            child1[-1] = fitness_values_child1['ValFitness']
            child1[-2] = fitness_values_child1['TrainFitness']

            target_function_parameters['weights'] = child2[:-2]
            fitness_values_child2 = target_function(**target_function_parameters)
            child2[-1] = fitness_values_child2['ValFitness']
            child2[-2] = fitness_values_child2['TrainFitness']

            # Check if the child's fitness is better than the worst individuals
            worst_idx0, worst_idx1 = worst_individuals[0], worst_individuals[1]
            if child1[-1] < offspring[worst_idx0, -1]:
                offspring[worst_idx0] = child1[:]

            if child2[-1] < offspring[worst_idx1, -1]:
                offspring[worst_idx1] = child2[:]

        else:
            # If no crossover, offspring are identical to parents
            offspring[parent_1_idx] = parent_1[:]
            offspring[parent_2_idx] = parent_2[:]

    return offspring



# Function: Mutation (Modified for Binary Representation)
def mutation(offspring,
             mutation_rate=0.1,
             target_function=target_function,
             target_function_parameters=None):
    for i in range(0, offspring.shape[0]):
        random_numbers = [
            random.uniform(0, 1) for _ in range(offspring.shape[1] - 2)
        ]
        offspring[i, :-2] = [
            1 - bit if probability < mutation_rate else bit
            for bit, probability in zip(offspring[i, :-2], random_numbers)
        ]

        target_function_parameters['weights'] = offspring[i, 0:-2]
        fitness_values = target_function(**target_function_parameters)
        offspring[i, -1] = fitness_values['ValFitness']
        offspring[i, -2] = fitness_values['TrainFitness']

    return offspring


############################################################################


# GA Function
def genetic_algorithm(population_size=5,
                      mutation_rate=0.1,
                      elite=0,
                      min_values=[-5, -5],
                      max_values=[5, 5],
                      crossover_rate=0.8,
                      generations=50,
                      target_function=target_function,
                      target_function_parameters=None,
                      verbose=True):
    count = 0
    fitness_values = []
    population = initial_population(population_size, min_values, max_values,
                                    target_function,
                                    target_function_parameters)
    fitness = fitness_function(population)
    elite_ind = np.copy(population[population[:, -1].argsort()][0, :])
    while (count <= generations):
        if (verbose == True):
            print('Generation = ', count, ' f(x) = ', elite_ind[-1])
        offspring = breeding(population, fitness, crossover_rate, elite,
                             target_function, target_function_parameters)
        population = mutation(offspring, mutation_rate, target_function,
                              target_function_parameters)
        fitness = fitness_function(population)
        value = np.copy(population[population[:, -1].argsort()][0, :])
        if (elite_ind[-1] > value[-1]):
            elite_ind = np.copy(value)
        count = count + 1
        fitness_values.append({
            'ValFitness': elite_ind[-1],
            'TrainFitness': elite_ind[-2]
        })

    return elite_ind, fitness_values


############################################################################
