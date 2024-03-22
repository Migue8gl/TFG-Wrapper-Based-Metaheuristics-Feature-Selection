# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Dragonfly Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

# Required Libraries
import random

import numpy as np
from scipy.special import gamma


# Function
def target_function():
    return


# Function: Initialize Variables
def initial_variables(size=5,
                      min_values=[-5, -5],
                      max_values=[5, 5],
                      target_function=target_function,
                      target_function_parameters=None):
    position = np.zeros((size, len(min_values) + 4))
    for i in range(0, size):
        for j in range(0, len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        target_function_parameters['weights'] = position[i, :-4]
        fitness = target_function(**target_function_parameters)
        position[i, -1] = fitness['fitness']
        position[i, -2] = fitness['accuracy']
        position[i, -3] = fitness['selected_features']
        position[i, -4] = fitness['selected_rate']
    return position


# Transfer functions S-Shaped
def s_shaped_transfer_function(x):
    threshold = np.random.random()
    return 1 if sigmoid(x) > threshold else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Transfer functions V-Shaped
def v_shaped_transfer_function(x, delta_x):
    threshold = np.random.random()
    return 1 - x if hyperbolic(delta_x) > threshold else x


def hyperbolic(x):
    return np.abs(x / np.sqrt(x**2 + 1))


# Function: Distance Calculations
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))


# Function: Levy Distribution
def levy_flight(d, beta, sigma):
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v)**(1 / beta)
    return 0.01 * step


# Function: Update Food and Enemy Positions
def update_food_enemy(dragonflies, food_pos, enemy_pos):
    best_food_idx = np.argmin(dragonflies[:, -1])
    if dragonflies[best_food_idx, -1] < food_pos[0, -1]:
        food_pos[0, :] = dragonflies[best_food_idx, :]
    worst_enemy_idx = np.argmax(dragonflies[:, -1])
    if dragonflies[worst_enemy_idx, -1] > enemy_pos[0, -1]:
        enemy_pos[0, :] = dragonflies[worst_enemy_idx, :]
    return food_pos, enemy_pos


# Function: Update Search Matrices
def update_position(a, c, f, e, s, w, r, beta, sigma, enemy_pos, food_pos,
                    delta_max, dragonflies, deltaflies, min_values, max_values,
                    target_function, target_function_parameters, binary):
    for i in range(dragonflies.shape[0]):
        neighbours_delta, neighbours_dragon = [], []
        for j in range(dragonflies.shape[0]):
            dist = euclidean_distance(dragonflies[i, :-4], dragonflies[j, :-4])
            if (dist > 0).all() and (dist <= r).all():
                neighbours_delta.append(deltaflies[j, :-4])
                neighbours_dragon.append(dragonflies[j, :-4])
        A = np.mean(neighbours_delta,
                    axis=0) if neighbours_delta else deltaflies[i, :-4]
        C = np.mean(neighbours_dragon, axis=0) - dragonflies[
            i, :-4] if neighbours_dragon else np.zeros(len(min_values))
        S = -np.sum(neighbours_dragon - dragonflies[i, :-4],
                    axis=0) if neighbours_dragon else np.zeros(len(min_values))
        dist_f = euclidean_distance(dragonflies[i, :-4], food_pos[0, :-4])
        dist_e = euclidean_distance(dragonflies[i, :-4], enemy_pos[0, :-4])
        F = food_pos[0, :-4] - dragonflies[i, :-4] if (
            dist_f <= r).all() else np.zeros(len(min_values))
        E = enemy_pos[0, :-4] if (dist_e <= r).all() else np.zeros(
            len(min_values))
        for k in range(len(min_values)):
            if (dist_f > r).all():
                if len(neighbours_dragon) > 1:
                    deltaflies[i,
                               k] = w * deltaflies[i, k] + np.random.rand() * (
                                   a * A[k] + c * C[k] + s * S[k])
                else:
                    dragonflies[i, :-4] = dragonflies[i, :-4] + levy_flight(
                        len(min_values), beta, sigma) * dragonflies[i, :-4]
                    deltaflies[i, k] = np.clip(deltaflies[i, k], min_values[k],
                                               max_values[k])
                    break
            else:
                deltaflies[i, k] = (a * A[k] + c * C[k] + s * S[k] + f * F[k] +
                                    e * E[k]) + w * deltaflies[i, k]
            deltaflies[i, k] = np.clip(deltaflies[i, k], -delta_max[k],
                                       delta_max[k])
            if binary == 's':
                dragonflies[i, k] = s_shaped_transfer_function(deltaflies[i,
                                                                          k])
            elif binary == 'v':
                dragonflies[i, k] = v_shaped_transfer_function(
                    dragonflies[i, k], deltaflies[i, k])
            else:
                dragonflies[i, k] = dragonflies[i, k] + deltaflies[i, k]
            dragonflies[i, k] = np.clip(dragonflies[i, k], min_values[k],
                                        max_values[k])
        target_function_parameters['weights'] = dragonflies[i, :-4]
        fitness = target_function(**target_function_parameters)
        dragonflies[i, -1] = fitness['fitness']
        dragonflies[i, -2] = fitness['accuracy']
        dragonflies[i, -3] = fitness['selected_features']
        dragonflies[i, -4] = fitness['selected_rate']
    food_pos, enemy_pos = update_food_enemy(dragonflies, food_pos, enemy_pos)
    best_dragon = np.copy(food_pos[food_pos[:, -1].argsort()][0, :])
    return enemy_pos, food_pos, dragonflies, deltaflies, best_dragon


# DA Function
def dragonfly_algorithm(size=3,
                        min_values=[-5, -5],
                        max_values=[5, 5],
                        generations=50,
                        target_function=target_function,
                        verbose=True,
                        binary='x',
                        target_function_parameters=None):
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    delta_max = (max_values - min_values) / 10
    food_pos = initial_variables(1, min_values, max_values, target_function,
                                 target_function_parameters)
    enemy_pos = initial_variables(1, min_values, max_values, target_function,
                                  target_function_parameters)
    dragonflies = initial_variables(size, min_values, max_values,
                                    target_function,
                                    target_function_parameters)
    deltaflies = initial_variables(size, min_values, max_values,
                                   target_function, target_function_parameters)
    beta = 3 / 2  # TODO probar y estudiar valores de beta y su función con la fase de explotación
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma(
        (1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    count = 0
    fitness_values = []
    for i in range(0, dragonflies.shape[0]):
        if dragonflies[i, -1] > enemy_pos[0, -1]:
            for j in range(0, dragonflies.shape[1]):
                enemy_pos[0, j] = dragonflies[i, j]
    best_dragon = np.copy(food_pos[food_pos[:, -1].argsort()][0, :])
    while count <= generations:
        if verbose:
            print('Generation: ', count, ' f(x) = ', best_dragon[-1])
        r = (max_values - min_values) / 4 + (
            (max_values - min_values) * count / generations * 2)
        w = 0.9 - count * ((0.9 - 0.4) / generations)
        my_c = 0.1 - count * ((0.1 - 0) / (generations / 2))
        my_c = np.max(my_c, 0)
        s = 2 * np.random.rand() * my_c  # Seperation Weight
        a = 2 * np.random.rand() * my_c  # Alignment Weight
        c = 2 * np.random.rand() * my_c  # Cohesion Weight
        f = 2 * np.random.rand()  # Food Attraction Weight
        e = my_c
        food_pos, enemy_pos = update_food_enemy(dragonflies, food_pos,
                                                enemy_pos)
        enemy_pos, food_pos, dragonflies, deltaflies, best_dragon = update_position(
            a, c, f, e, s, w, r, beta, sigma, enemy_pos, food_pos, delta_max,
            dragonflies, deltaflies, min_values, max_values, target_function,
            target_function_parameters, binary)
        count += 1
        fitness_values.append({
            'fitness': best_dragon[-1],
            'accuracy': best_dragon[-2],
            'selected_features': best_dragon[-3],
            'selected_rate': best_dragon[-4]
        })
    return best_dragon, fitness_values


############################################################################
