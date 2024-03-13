import numpy as np


# TODO add parameters
# Function
def target_function():
    return


# Function: Initialize Ants
def initial_ants(colony_size=5):
    return [[[], 1, 1] for _ in range(colony_size)]


# Function: Initialize Pheromones
def initial_pheromone_graph(n_features, initial_pheromone):
    pheromone = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
                pheromone[i, j] = initial_pheromone
    return pheromone

# Function: Ant Build Subset
def ant_build_subset(ant, n_features, feature_pheromone, alpha):
    for node_i in range(n_features):
        p = np.zeros(n_features)
        p_num = np.zeros(n_features)
        for node_j in range(n_features):
            tau = feature_pheromone[node_i][node_j]
            p_num[node_j] = (tau**alpha)
        den = np.sum(p_num)
        p = p_num / den
        r = np.random.rand()
        if r <= p[node_i]:
            ant[0].append(1)
        else:
            ant[0].append(0)


# Function: Update Pheromones
# TODO update function
def update_pheromones(best_ant, feature_pheromone, evaporation_rate,
                      Q_constant):
    delta_pheromones = np.zeros(shape=feature_pheromone.shape)
    for feature in best_ant[0]:
        delta_pheromones[feature] += Q_constant / (best_ant[1])
    feature_pheromone = (
        1 - evaporation_rate) * feature_pheromone + delta_pheromones
    return np.clip(feature_pheromone, 0.1, 6)


# ACO Function
def ant_colony_optimization(n_ants=20,
                            iterations=100,
                            n_features=10,
                            alpha=1,
                            q=1,
                            initial_pheromone=1.0,
                            evaporation_rate=0.1,
                            min_values=[-5, -5],
                            max_values=[5, 5],
                            target_function_parameters=None,
                            target_function=target_function,
                            verbose=True):
    ants = initial_ants(n_ants)
    best_ant = ants[0]
    feature_pheromone = initial_pheromone_graph(n_features, initial_pheromone)
    fitness_values = []
    for count in range(iterations):
        if (verbose):
            print('Iteration = ', count, ' f(x) = ', best_ant[1])
        for ant in ants:
            # Local search for each ant
            ant_build_subset(ant, n_features, feature_pheromone, alpha)
            target_function_parameters['weights'] = np.array(ant[0])
            fitness = target_function(**target_function_parameters)
            ant[1] = fitness['validation']['fitness']
            ant[2] = fitness['training']['fitness']
        best_local_ant = min(ants, key=lambda ant: ant[1])
        fitness_values.append({
            'val_fitness': best_ant[1],
            'train_fitness': best_ant[2]
        })
        if best_local_ant[1] < best_ant[1]:
            best_ant = best_local_ant

        feature_pheromone = update_pheromones(best_ant, feature_pheromone,
                                              evaporation_rate, q)
        ants = initial_ants(n_ants)

    best_ant[0].append(best_ant[1])
    best_ant[0].append(best_ant[2])
    best_ant = np.array(best_ant[0])
    
    return best_ant, fitness_values
