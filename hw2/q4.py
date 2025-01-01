import numpy as np
import matplotlib.pyplot as plt

# Process parameters
K = 1.0           # Gain
tau = 10.0        # Time constant
tau_DT = 1.0      # Dead time
dt = 0.1          # Simulation time step
time_end = 50     # End time for the simulation
t = np.arange(0, time_end, dt)

# Set-point for the system to follow
setpoint = np.ones_like(t)

# Discretized FOPTD process model
def simulate_system(kp, ki, kd, K, tau, tau_DT, t, setpoint):
    # Initialize variables
    n = len(t)
    error = np.zeros(n)
    integral = 0
    derivative = 0
    output = np.zeros(n)
    plant_output = np.zeros(n)
    
    # Loop through each time step
    for i in range(1, n):
        error[i] = setpoint[i] - plant_output[i-1]
        integral += error[i] * dt
        if i > 1:
            derivative = (error[i] - error[i-1]) / dt
        
        # PID Control
        control_signal = kp * error[i] + ki * integral + kd * derivative

        # FOPTD Process with dead time
        if i * dt > tau_DT:
            plant_output[i] = (K / tau) * control_signal * dt + (1 - dt / tau) * plant_output[i-1]
    
    return plant_output

# Define performance indices to minimize
def calculate_performance_indices(response, setpoint, t):
    # Calculate rise time
    rise_time = t[np.where(response >= 0.9 * setpoint[-1])[0][0]] if max(response) >= 0.9 * setpoint[-1] else np.inf
    # Calculate overshoot
    overshoot = (max(response) - setpoint[-1]) / setpoint[-1] * 100
    # Calculate settling time
    settling_time = t[np.where(abs(response - setpoint[-1]) <= 0.02 * setpoint[-1])[0][-1]] if any(abs(response - setpoint[-1]) <= 0.02 * setpoint[-1]) else np.inf
    # Calculate SSE
    sse = np.sum((setpoint - response) ** 2) * dt
    return rise_time, overshoot, settling_time, sse

# Objective function for PSO
def objective_function(params):
    kp, ki, kd = params
    response = simulate_system(kp, ki, kd, K, tau, tau_DT, t, setpoint)
    rise_time, overshoot, settling_time, sse = calculate_performance_indices(response, setpoint, t)
    # Weighted sum of performance indices
    cost = 0.25 * rise_time + 0.25 * settling_time + 0.25 * overshoot + 0.25 * sse
    return cost

# Run PSO to optimize PID parameters
lb = [0, 0, 0]   # Lower bounds for [kp, ki, kd]
ub = [10, 10, 10] # Upper bounds for [kp, ki, kd]

iterations = 1000
num_particles = 30
c_personal = 0.5
c_social = 0.5
num_dimension = 3
inertia_weight = 0.5
positions = np.random.uniform(0, 10, (num_particles, num_dimension))
velocities = np.random.uniform(-1, 1, (num_particles, num_dimension))

p_best_positions = np.copy(positions)
p_best_scores = np.array([objective_function(p) for p in positions])
g_best_position = p_best_positions[np.argmin(p_best_scores)]
g_best_score = np.min(p_best_scores)

for iteration in range(iterations):
    for i in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()
        p_vel = c_personal*r1*(p_best_positions[i]-positions[i])
        s_vel = c_social*r2*(g_best_position-positions[i])
        velocities[i] = inertia_weight*velocities[i] + p_vel + s_vel
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

# Display optimal PID parameters and corresponding cost
print("Optimal PID parameters (kp, ki, kd):", g_best_position)
print("Optimal cost:", g_best_score)

# Simulate and plot the response with optimized PID parameters
kp, ki, kd = g_best_position
response = simulate_system(kp, ki, kd, K, tau, tau_DT, t, setpoint)

# Plot the response
plt.plot(t, setpoint, 'r--', label="Setpoint")
plt.plot(t, response, 'b', label="Response")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("System Response with Optimized PID")
plt.legend()
plt.grid()
plt.show()
