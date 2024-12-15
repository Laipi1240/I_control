import numpy as np

# Define the Rosenbrock function
def rosenbrock(x):
    x1, x2 = x
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

# Particle Swarm Optimization function
def particle_swarm_optimization(func, dim, bounds, num_particles, max_iter, w=0.5, c1=1.5, c2=1.5):
    # Initialize particles
    particles = np.random.uniform(low=bounds[0], high=bounds[1], size=(num_particles, dim))
    velocities = np.random.uniform(low=-1, high=1, size=(num_particles, dim))
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([func(p) for p in particles])
    
    # Global best
    global_best_index = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_score = personal_best_scores[global_best_index]
    update_times=0
    
    for s in range(max_iter):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - particles[i])
                + c2 * r2 * (global_best_position - particles[i])
            )
            # Update position
            particles[i] += velocities[i]
            # Enforce boundary constraints
            particles[i] = np.clip(particles[i], bounds[0], bounds[1])
            # Evaluate fitness
            score = func(particles[i])
            # Update personal best
            if score < personal_best_scores[i]:
                personal_best_positions[i] = particles[i]
                personal_best_scores[i] = score
        # Update global best
        best_index = np.argmin(personal_best_scores)
        if global_best_index != best_index:
            update_times += 1
            global_best_index = best_index
            global_best_position = personal_best_positions[global_best_index].copy()
            global_best_score = personal_best_scores[global_best_index]
        
    return global_best_position, global_best_score, update_times

def multiple_test(times, func, dim, bounds, num_particles, max_iter):
    avg_b_pos = np.zeros(shape=dim)
    avg_b_score = 0.0
    avg_update_times = 0.0
    for _ in range(int(times)):
        best_position, best_score, update_times = particle_swarm_optimization(func, dim, bounds, num_particles, max_iter)
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
num_particles = 100  # Number of particles
max_iter = 100  # Maximum iterations
evaluate_times = 20.0

# Run PSO
# best_position, best_score, update_times = particle_swarm_optimization(rosenbrock, dim, bounds, num_particles, max_iter)
avg_b_pos, avg_b_score, avg_update_times = multiple_test(evaluate_times, rosenbrock, dim, bounds, num_particles, max_iter)

print("evaluate times: ", int(evaluate_times))
print("average best position: ", avg_b_pos)
print("average best score: ", avg_b_score)
print("average update times: ", avg_update_times)
