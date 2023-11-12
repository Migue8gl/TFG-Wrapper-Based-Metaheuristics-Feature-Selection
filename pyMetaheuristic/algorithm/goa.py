############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Grasshopper Optimization Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
import random

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_position(grasshoppers = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((grasshoppers, len(min_values)+1))
    for i in range(0, grasshoppers):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# Transfer functions S-Shaped
def sigmoid_threshold(x):
    threshold = np.random.rand()
    return 1 if sigmoid(x) > threshold else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

############################################################################

# Transfer functions V-Shaped
def hiperbolic_tan_threshold(delta_x, x):
    threshold = np.random.rand()
    return 1-delta_x if hiperbolic_tan(x) > threshold else delta_x

def hiperbolic_tan(x):
    return np.abs(np.tanh(x))

############################################################################

# Function: S
def s_function(r, F, L):
    s = F*np.exp(-r/L) - np.exp(-r)
    return s

# Function: Distance Matrix
def build_distance_matrix(position):
   a = position[:,:-1]
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Update Position
def update_position(position, best_position, min_values, max_values, C, F, L, target_function, binary):
    sum_grass       = 0
    distance_matrix = build_distance_matrix(position)
    distance_matrix = 2*(distance_matrix - np.min(distance_matrix))/(np.ptp(distance_matrix)+0.00000001) + 1
    np.fill_diagonal(distance_matrix , 0)
    for i in range(0, position.shape[0]):
        for j in range(0, len(min_values)):
            for k in range(0, position.shape[0]):
                if (k != i):
                    sum_grass = sum_grass + C * ((max_values[j] - min_values[j])/2) * s_function(distance_matrix[k, i], F, L) * ((position[k, j] - position[i, j])/distance_matrix[k, i])
            if binary == 's':
                position[i, j] = sigmoid_threshold(np.clip(C*sum_grass, min_values[j], max_values[j]))
            elif binary == 'v':
                position[i, j] = hiperbolic_tan_threshold(np.clip(C*sum_grass + best_position[0, j], min_values[j], max_values[j]), np.clip(C*sum_grass, min_values[j], max_values[j]))
            else:
                position[i, j] = np.clip(C*sum_grass + best_position[0, j], min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# GOA Function
def grasshopper_optimization_algorithm(grasshoppers = 5, min_values = [-5,-5], max_values = [5,5], c_min = 0.00004, c_max = 1, iterations = 1000, F = 0.5, L = 1.5, target_function = target_function, binary = 's', verbose = True):
    count         = 0
    position      = initial_position(grasshoppers, min_values, max_values, target_function)
    best_position = np.copy(position[np.argmin(position[:,-1]),:].reshape(1,-1))   

    # Lists to store fitness values
    fitness_values = []

    while (count <= iterations): 
        if (verbose == True):
            print('Iteration = ', count,  ' f(x) = ', best_position[0, -1])
            # Store fitness value and iteration number
            fitness_values.append(best_position[0, -1])
        C = c_max - count*( (c_max - c_min)/iterations)
        position = update_position(position, best_position, min_values, max_values, C, F, L, target_function = target_function , binary = binary)
        if (np.amin(position[:,-1]) < best_position[0,-1]):
            best_position = np.copy(position[np.argmin(position[:,-1]),:].reshape(1,-1))  
        count    = count + 1
    return best_position.flatten(), fitness_values

############################################################################
