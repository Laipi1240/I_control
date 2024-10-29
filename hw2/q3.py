import numpy as np

def objective_function(x):
    a = 20
    b = 0.2
    c = 2 * np.math.pi
    return -1 * a * np.exp(-1*b*np.math.sqrt(np.mean(x**2))) -np.exp(np.mean(np.cos(c*x))) + a + np.exp(1)

iterations = 1000

def inertia_weight(n_iter):
    max_inertia_weight = 1.5
    min_inertia_weight = 0.5
    return max_inertia_weight-(max_inertia_weight-min_inertia_weight)*n_iter/iterations

def vel_threshold(vel):
    pass

num_particles = 30
num_dimensions = 2
c_personal = 0.5
c_social = 0.5

positions = np.random.uniform(-10, 10, (num_particles, num_dimensions))
velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))

p_best_positions = np.copy(positions)
p_best_scores = np.array([objective_function(p) for p in positions])
g_best_position = p_best_positions[np.argmin(p_best_scores)]
g_best_score = np.min(p_best_scores)

for iteration in range(iterations):
    for i in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()
        p_vel = c_personal*r1*(p_best_positions[i]-positions[i])
        s_vel = c_social*r2*(g_best_position-positions[i])
        velocities[i] = inertia_weight(iteration)*velocities[i] + p_vel + s_vel

        positions[i] += velocities[i]

        score = objective_function(positions[i])
        if score < p_best_scores[i]:
            p_best_scores[i] = score
            p_best_positions[i] = positions[i]

    best_particle_index = np.argmin(p_best_scores)
    if p_best_scores[best_particle_index] < g_best_score:
        g_best_score = p_best_scores[best_particle_index]
        g_best_position = p_best_positions[best_particle_index]

    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Best Score: {g_best_score}")

print("\nBest solution found:")
print("Position:", g_best_position)
print("Objective value:", g_best_score)