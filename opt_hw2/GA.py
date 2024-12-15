import numpy as np

# Define the Rosenbrock function
def rosenbrock(x):
    x1, x2 = x
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

# Genetic Algorithm function
def genetic_algorithm(
    func, dim, bounds, population_size, generations, mutation_rate=0.01, crossover_rate=0.7
):
    # Initialize the population randomly within bounds
    population = np.random.uniform(low=bounds[0], high=bounds[1], size=(population_size, dim))
    fitness = np.array([func(ind) for ind in population])  # Evaluate fitness of the initial population
    best_idx = np.argmin(fitness)
    update_times = 0

    # Helper functions for selection, crossover, and mutation
    def select_parent():
        """Tournament selection."""
        competitors = np.random.choice(population_size, size=5, replace=False)
        best_idx = competitors[np.argmin(fitness[competitors])]
        return population[best_idx]

    def crossover(parent1, parent2):
        """Single-point crossover."""
        if np.random.rand() < crossover_rate:
            point = np.random.randint(1, dim)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        return child1, child2

    def mutate(individual):
        """Random mutation within bounds."""
        for i in range(dim):
            if np.random.rand() < mutation_rate:
                individual[i] += np.random.uniform(-1, 1)
                individual[i] = np.clip(individual[i], bounds[0], bounds[1])
        return individual

    # Run the GA for a given number of generations
    for _ in range(generations):
        # Generate the next generation
        new_population = []
        while len(new_population) < population_size:
            # Select parents
            parent1 = select_parent()
            parent2 = select_parent()
            # Perform crossover
            child1, child2 = crossover(parent1, parent2)
            # Perform mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        # Ensure population size matches
        population = np.array(new_population[:population_size])
        fitness = np.array([func(ind) for ind in population])  # Evaluate fitness
        update_best_idx = np.argmin(fitness)
        if best_idx != update_best_idx:
            update_times += 1
            best_idx = update_best_idx

    # Find the best solution
    best_position = population[best_idx]
    best_score = fitness[best_idx]
    return best_position, best_score, update_times

def multiple_test(times, func, dim, bounds, population_size, generations, mutation_rate, crossover_rate):
    avg_b_pos = np.zeros(shape=dim)
    avg_b_score = 0.0
    avg_update_times = 0.0
    for _ in range(int(times)):
        best_position, best_score, update_times = genetic_algorithm(
            func, dim, bounds, population_size, generations, mutation_rate, crossover_rate
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
population_size = 100  # Number of individuals in the population
generations = 100  # Number of generations
mutation_rate = 0.01  # Mutation rate
crossover_rate = 0.8  # Crossover rate
evaluate_times = 20.0

# Run GA
# best_position, best_score = genetic_algorithm(
#     rosenbrock, dim, bounds, population_size, generations, mutation_rate, crossover_rate
# )
avg_b_pos, avg_b_score, avg_update_times = multiple_test(evaluate_times, rosenbrock, dim, bounds, population_size, generations, mutation_rate, crossover_rate)

print("evaluate times: ", int(evaluate_times))
print("average best position: ", avg_b_pos)
print("average best score: ", avg_b_score)
print("average update times: ", avg_update_times)