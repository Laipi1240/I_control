import numpy as np
import scipy.special as sp

# Define the Rosenbrock function
def rosenbrock(x):
    x1, x2 = x
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

# Levy flight step
def levy_flight(Lambda):
    numerator = sp.gamma(1 +Lambda) * np.sin(np.pi * Lambda / 2)
    denominator = sp.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)
    sigma_u = (numerator / denominator)**(1 / Lambda)

    u = np.random.normal(0, 1) * sigma_u
    v = np.random.normal(0, 1)
    step = u / abs(v)**(1 / Lambda)
    return step

# Cuckoo Search Algorithm
def cuckoo_search(func, dim, bounds, n_nests=25, max_iter=100, pa=0.25, alpha=0.01, levy_Lambda=1.5):
    # Initialize nests randomly
    nests = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_nests, dim))
    fitness = np.array([func(nest) for nest in nests])  # Fitness of the nests

    # Find the best nest
    best_nest_idx = np.argmin(fitness)
    best_nest = nests[best_nest_idx].copy()
    best_score = fitness[best_nest_idx]
    update_times = 0

    for _ in range(max_iter):
        # Generate a new solution by Levy flight for each cuckoo
        for i in range(n_nests):
            step_size = np.random.normal(0, 1) * alpha * levy_flight(levy_Lambda) * (nests[i] - best_nest)
            new_nest = nests[i] + step_size
            new_nest = np.clip(new_nest, bounds[0], bounds[1])  # Apply boundary constraints
            new_fitness = func(new_nest)

            # Replace nest if the new solution is better
            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness

        # Abandon a fraction of the nests and generate new ones
        abandoned = np.random.rand(n_nests) < pa
        for i in range(n_nests):
            if abandoned[i]:
                nests[i] = np.random.uniform(low=bounds[0], high=bounds[1], size=dim)
                fitness[i] = func(nests[i])  # Update fitness for the new nest

        # Find the current best nest
        nest_idx = np.argmin(fitness)
        if best_nest_idx != nest_idx:
            update_times += 1
            best_nest_idx = nest_idx
        best_nest = nests[best_nest_idx].copy()
        best_score = fitness[best_nest_idx]

    return best_nest, best_score, update_times

def multiple_test(times, func, dim, bounds, n_nests, max_iter, pa, alpha, levy_Lambda):
    avg_b_pos = np.zeros(shape=dim)
    avg_b_score = 0.0
    avg_update_times = 0.0
    for _ in range(int(times)):
        best_position, best_score, update_times = cuckoo_search(
            func, dim, bounds, n_nests, max_iter, pa, alpha, levy_Lambda
        )
        avg_b_pos += best_position
        avg_b_score += best_score
        avg_update_times += update_times
    avg_b_pos /= times
    avg_b_score /= times
    avg_update_times /= times
    return avg_b_pos, avg_b_score, avg_update_times

# Parameters
dim = 2  # Dimensionality of the problem
bounds = [-5, 5]  # Search space bounds
n_nests = 100  # Number of nests (population size)
max_iter = 1000  # Maximum number of iterations
pa = 0.05  # Probability of abandoning a nest
alpha = 0.01  # Step size for Levy flight
levy_Lambda = 1.5  # Levy flight parameter
evaluate_times = 20.0

# Run Cuckoo Search
# best_position, best_score = cuckoo_search(rosenbrock, dim, bounds, n_nests, max_iter, pa, alpha, levy_Lambda)
avg_b_pos, avg_b_score, avg_update_times = multiple_test(evaluate_times, rosenbrock, dim, bounds, n_nests, max_iter, pa, alpha, levy_Lambda)

print("evaluate times: ", int(evaluate_times))
print("average best position: ", avg_b_pos)
print("average best score: ", avg_b_score)
print("average update times: ", avg_update_times)
