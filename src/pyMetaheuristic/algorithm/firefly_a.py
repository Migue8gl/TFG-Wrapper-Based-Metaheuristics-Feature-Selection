############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Firefly Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np
import random
import math
import os

############################################################################


# Function
def target_function():
    return


############################################################################

# Transfer functions S-Shaped


def s_shaped_transfer_function(x):
    threshold = np.random.random()
    return 1 if sigmoid(x) > threshold else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Transfer functions V-Shaped


def v_shaped_transfer_function(x):
    threshold = np.random.random()
    return 1 - x if hyperbolic(x) > threshold else x


def hyperbolic(x):
    return np.abs(x / np.sqrt(x**2 + 1))


############################################################################


# Function: Initialize Variables
def initial_fireflies(swarm_size=3,
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
        position[i, -1] = fitness['validation']['fitness']
        position[i, -2] = fitness['training']['fitness']
    return position


############################################################################


# Function: Distance Calculations
def euclidean_distance(x, y):
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance
    return distance**(1 / 2)


############################################################################


# Function: Beta Value
def beta_value(x, y, gama=1, beta_0=1):
    rij = euclidean_distance(x, y)
    beta = beta_0 * math.exp(-gama * (rij)**2)
    return beta


# Function: Ligth Intensity
def ligth_value(light_0, x, y, gama=1):
    rij = euclidean_distance(x, y)
    light = light_0 * math.exp(-gama * (rij)**2)
    return light


# Function: Update Position
def update_position(position,
                    x,
                    y,
                    alpha_0=0.2,
                    beta_0=1,
                    gama=1,
                    firefly=0,
                    min_values=[-5, -5],
                    max_values=[5, 5],
                    target_function=target_function,
                    target_function_parameters=None,
                    binary='x'):
    for j in range(0, len(x) - 2):
        epson = int.from_bytes(os.urandom(8), byteorder="big") / (
            (1 << 64) - 1) - (1 / 2)
        if binary == 's':
            position[firefly, j] = s_shaped_transfer_function(
                x[j] + beta_value(x, y, gama=gama, beta_0=beta_0) *
                (y[j] - x[j]) + alpha_0 * epson)
        elif binary == 'v':
            position[firefly, j] = v_shaped_transfer_function(
                x[j] + beta_value(x, y, gama=gama, beta_0=beta_0) *
                (y[j] - x[j]) + alpha_0 * epson)
        else:
            position[firefly, j] = np.clip(
                (x[j] + beta_value(x, y, gama=gama, beta_0=beta_0) *
                 (y[j] - x[j]) + alpha_0 * epson), min_values[j],
                max_values[j])
    target_function_parameters['weights'] = position[firefly, :-2]
    fitness = target_function(**target_function_parameters)
    position[firefly, -1] = fitness['validation']['fitness']
    position[firefly, -2] = fitness['training']['fitness']

    return position


############################################################################


# FA Function
def firefly_algorithm(swarm_size=3,
                      min_values=[-5, -5],
                      max_values=[5, 5],
                      generations=50,
                      alpha_0=0.2,
                      beta_0=1,
                      gama=1,
                      target_function=target_function,
                      binary='x',
                      target_function_parameters=None,
                      verbose=True):
    count = 0
    fitness_values = []
    position = initial_fireflies(
        swarm_size=swarm_size,
        min_values=min_values,
        max_values=max_values,
        target_function=target_function,
        target_function_parameters=target_function_parameters)
    while (count <= generations):
        best_firefly = np.copy(position[position[:, -1].argsort()][0, :])
        if (verbose):
            print('Generation: ', count, ' f(x) = ', best_firefly[-1])
        for i in range(0, swarm_size):
            for j in range(0, swarm_size):
                if (i != j):
                    firefly_i = np.copy(position[i, 0:position.shape[1] - 1])
                    firefly_j = np.copy(position[j, 0:position.shape[1] - 1])
                    ligth_i = ligth_value(position[i, -1],
                                          firefly_i,
                                          firefly_j,
                                          gama=gama)
                    ligth_j = ligth_value(position[j, -1],
                                          firefly_i,
                                          firefly_j,
                                          gama=gama)
                    if (ligth_i > ligth_j):
                        position = update_position(position, firefly_i,
                                                   firefly_j, alpha_0, beta_0,
                                                   gama, i, min_values,
                                                   max_values, target_function,
                                                   target_function_parameters,
                                                   binary)
        count = count + 1
        fitness_values.append({
            'val_fitness': best_firefly[-1],
            'train_fitness': best_firefly[-2]
        })
    best_firefly = np.copy(position[position[:, -1].argsort()][0, :])
    return best_firefly, fitness_values


############################################################################
