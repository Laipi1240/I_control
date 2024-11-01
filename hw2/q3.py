import numpy as np
from q3_util import levy, ackley
import matplotlib.pyplot as plt

iterations = 1000
lr_list = [0.1, 0.3, 0.5, 0.7]
power_list = [5, 10, 15, 20]
# def inertia_weight(n_iter):
#     max_inertia_weight = 1.5
#     min_inertia_weight = 0.5
#     return max_inertia_weight-(max_inertia_weight-min_inertia_weight)*n_iter/iterations

def vel_threshold(vel):
    pass

MSE_list = []
MSE_var_list = []
num_particles = 30
c_personal = 0.5
c_social = 0.5
i = 0
j = 0
for lr in lr_list:
    inertia_weight = lr
    MSE_sublist = []
    for dim in power_list:
        num_dimensions = dim
        opt_pos = np.ones(num_dimensions)

        positions = np.random.uniform(-10, 10, (num_particles, num_dimensions))
        velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))

        p_best_positions = np.copy(positions)
        p_best_scores = np.array([levy(p) for p in positions])
        g_best_position = p_best_positions[np.argmin(p_best_scores)]
        g_best_score = np.min(p_best_scores)

        MSE_per_iteration = []
        for iteration in range(iterations):
            for i in range(num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                p_vel = c_personal*r1*(p_best_positions[i]-positions[i])
                s_vel = c_social*r2*(g_best_position-positions[i])
                velocities[i] = inertia_weight*velocities[i] + p_vel + s_vel

                positions[i] += velocities[i]

                score = levy(positions[i])
                if score < p_best_scores[i]:
                    p_best_scores[i] = score
                    p_best_positions[i] = positions[i]

            best_particle_index = np.argmin(p_best_scores)
            if p_best_scores[best_particle_index] < g_best_score:
                g_best_score = p_best_scores[best_particle_index]
                g_best_position = p_best_positions[best_particle_index]

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best Score: {g_best_score}")
            MSE_per_iteration.append(np.mean((g_best_position - opt_pos)**2))
        print("\nBest solution found:")
        print("Position:", g_best_position)
        print("Objective value:", g_best_score)
        # MSE_list.append(np.mean((g_best_position - opt_pos)**2  ))
        MSE_sublist.append(MSE_per_iteration)
        MSE_var_list.append(np.var(MSE_per_iteration))
        j += 1
    MSE_list.append(MSE_sublist)
    i += 1

# plt.scatter(np.tile(power_list,4), MSE_list, color='green')
# plt.xlabel("power")
# plt.ylabel("MSE")
# plt.title("power vs. MSE")
# plt.grid()
# plt.show()
print(len(MSE_list[0]))
for i in range(len(lr_list)):
    for j in range(len(power_list)):
        plt.plot(MSE_list[i][j], label=f"lr={lr_list[i]}, power={power_list[j]}")
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.title("epoch vs. MSE")
plt.legend(loc='upper right')
plt.grid()
plt.show()