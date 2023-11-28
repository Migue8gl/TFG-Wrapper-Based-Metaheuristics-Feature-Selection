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
def initial_position(grasshoppers = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function, target_function_parameters = None):
    position = np.zeros((grasshoppers, len(min_values)+2))
    for i in range(0, grasshoppers):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        target_function_parameters['weights'] = position[i,0:position.shape[1]-2]
        position[i,-1] = target_function(**target_function_parameters)['ValFitness']
    return position

############################################################################

# Transfer functions S-Shaped
def s_shaped_transfer_function(x):
    threshold = np.random.rand()
    return 1 if sigmoid(x) > threshold else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

############################################################################

# Transfer functions V-Shaped
def v_shaped_transfer_function(delta_x, x):
    threshold = np.random.rand()
    return 1-delta_x if hyperbolic_tan(x) > threshold else delta_x

def hyperbolic_tan(x):
    return np.abs(np.tanh(x))

############################################################################

# Function: S
def s_function(r, F, L):
    s = F*np.exp(-r/L) - np.exp(-r)
    return s

# Function: Distance Matrix
def build_distance_matrix(position):
   a = position[:,:-2]
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Update Position
def update_position(position, best_position, min_values, max_values, C, F, L, target_function, binary, target_function_parameters):
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
                position[i, j] = s_shaped_transfer_function(np.clip(C*sum_grass, min_values[j], max_values[j]))
            elif binary == 'v':
                position[i, j] = v_shaped_transfer_function(np.clip(C*sum_grass + best_position[0, j], min_values[j], max_values[j]), np.clip(C*sum_grass, min_values[j], max_values[j]))
            else:
                position[i, j] = np.clip(C*sum_grass + best_position[0, j], min_values[j], max_values[j])
        target_function_parameters['weights'] = position[i,0:position.shape[1]-2]
        fitness_values = target_function(**target_function_parameters)
        position[i, -1] = fitness_values['ValFitness']
        position[i, -2] = fitness_values['TrainFitness']
    return position

############################################################################

# GOA Function
def grasshopper_optimization_algorithm(grasshoppers = 5, min_values = [-5,-5], max_values = [5,5], c_min = 0.00004, c_max = 1, iterations = 1000, F = 0.5, L = 1.5, target_function = target_function, binary = 'x', verbose = True, target_function_parameters = None):
    count         = 0
    position      = initial_position(grasshoppers, min_values, max_values, target_function, target_function_parameters)
    best_position = np.copy(position[np.argmin(position[:,-1]),:].reshape(1,-1))   

    # Lists to store fitness values
    fitness_values = []

    while (count <= iterations): 
        C = c_max - count*( (c_max - c_min)/iterations)
        position = update_position(position, best_position, min_values, max_values, C, F, L, target_function = target_function , binary = binary, target_function_parameters = target_function_parameters)
        if (np.amin(position[:,-1]) < best_position[0,-1]):
            best_position = np.copy(position[np.argmin(position[:,-1]),:].reshape(1,-1))  
        count    = count + 1
        if (verbose == True):
            print('Iteration = ', count,  ' f(x) = ', best_position[0, -1])
        fitness_values.append({'ValFitness': best_position[0, -1], 'TrainFitness': best_position[0, -2]})
    return best_position.flatten(), fitness_values

############################################################################
