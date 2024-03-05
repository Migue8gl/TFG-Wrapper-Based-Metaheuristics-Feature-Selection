import numpy as np


# TODO add parameters
# Function
def target_function():
    return


# Function: Initialize Ants
def initial_ants(colony_size=5, n_features=10, binary=True):
    ants = []
    if binary:
        ants = [[[] * colony_size, 1, 1] for _ in range(colony_size)]
    else:
        for _ in range(colony_size):
            ants.append([[np.random.uniform(0, 1) for _ in range(n_features)],
                         1, 1])
    return ants


# Function: Initialize Pheromones
def initial_pheromone_graph(n_features, initial_pheromone, std_dev, binary):
    if not binary:
        qk = std_dev * n_features
        w = np.zeros((n_features))
        for i in range(n_features):
            w[i] = 1 / (qk * 2 * np.pi) * np.exp(
                -np.power(i, 2) /
                (2 * np.power(std_dev, 2) * np.power(n_features, 2)))
    else:
        w = np.array([[initial_pheromone] * n_features
                      for _ in range(n_features)])
    return w


# Gaussian Kernel (Probability Density Function)


def gaussian_kernel(x, w, means, std_deviations, epsilon=1e-10):
    std_deviations = np.maximum(std_deviations,
                                epsilon)  # Ensure std_deviations are not zero
    p = np.sum(w * 1 / (std_deviations * np.sqrt(2 * np.pi)) *
               np.exp(-((x - means)**2) / (2 * std_deviations**2)))
    return p


def roulette_wheel_selection(p):
    r = np.random.rand()
    C = np.cumsum(p)
    j = np.argmax(r <= C)
    return j


# Function: Ant Build Subset
def ant_build_subset(ant, n_features, feature_pheromone, alpha, sigma, binary):
    if not binary:
        for i in range(n_features):
            ant[0][i] = gaussian_kernel(
                ant[0][i], feature_pheromone[i],
                np.random.random_sample(size=n_features), sigma)
    else:
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
                            std_dev=0.3,
                            initial_pheromone=1.0,
                            evaporation_rate=0.1,
                            min_values=[-5, -5],
                            max_values=[5, 5],
                            target_function_parameters=None,
                            target_function=target_function,
                            verbose=True,
                            binary=True):
    ants = initial_ants(n_ants, n_features, binary)
    best_ant = ants[0]
    feature_pheromone = initial_pheromone_graph(n_features, initial_pheromone,
                                                std_dev, binary)
    fitness_values = []
    for count in range(iterations):
        if (verbose):
            print('Iteration = ', count, ' f(x) = ', best_ant[1])
        for ant in ants:
            # Local search for each ant
            sigma = np.zeros(n_features)
            if not binary:
                p = feature_pheromone / sum(feature_pheromone)
                # Find the index of the best
                ix_selected = roulette_wheel_selection(p)

                # Calculate sigma
                sigma_s = np.zeros(n_features)
                for i in range(n_features):
                    for j in range(len(ants)):
                        sigma_s[i] = np.sqrt(
                            np.power(ants[j][0][i] - ants[j][0][ix_selected],
                                     2))
                    sigma[i] = evaporation_rate / (n_features) * sigma_s[i]
            ant_build_subset(ant, n_features, feature_pheromone, alpha, sigma,
                             binary)
            target_function_parameters['weights'] = np.array(ant[0])
            fitness = target_function(**target_function_parameters)
            ant[1] = fitness['ValFitness']
            ant[2] = fitness['TrainFitness']
        best_local_ant = min(ants, key=lambda ant: ant[1])
        fitness_values.append({
            'ValFitness': best_ant[1],
            'TrainFitness': best_ant[2]
        })
        if best_local_ant[1] < best_ant[1]:
            best_ant = best_local_ant

        if binary:
            feature_pheromone = update_pheromones(best_ant, feature_pheromone,
                                                  evaporation_rate, q)
        ants = initial_ants(n_ants, n_features, binary)
    best_ant[0].append(best_ant[1])
    best_ant[0].append(best_ant[2])
    best_ant = np.array(best_ant[0])
    return best_ant, fitness_values
