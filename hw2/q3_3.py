import numpy as np
import matplotlib.pyplot as plt
from q3_util import ideal_square_wave, square_wave_fourier_series, cal_score

iterations = 100

# def inertia_weight(n_iter):
#     max_inertia_weight = 0.1
#     min_inertia_weight = 0.1
#     return max_inertia_weight-(max_inertia_weight-min_inertia_weight)*n_iter/iterations

def vel_threshold(vel):
    pass
lr_list = [0.5]
power_list = [20]
num_particles = 30
c_personal = 2
c_social = 2
t = np.linspace(-2*np.pi, 2*np.pi, 100)

MSE_list = []
MSE_var_list = []
MSE_his_list = []
i = 0
j = 0
for power in power_list:
    MSE_sublist = []
    for lr in lr_list:
        inertia_weight = lr
        num_order = power
        positions = np.random.uniform(-1, 1, (num_particles, 2*num_order+1))
        for i in range(num_particles):
            pos = np.zeros(2*num_order+1)
            for j in range(2*num_order+1):
                if j >= num_order+1:
                    if (j-num_order-1) % 2 ==0:
                        pos[j] = 4/((j-num_order)*np.pi)
                    else:
                        pos[j] = 0.0
                else:
                    pos[j] = 0.0
            positions[i] = pos

        velocities = np.random.uniform(-1, 1, (num_particles, 2*num_order+1))
        p_best_positions = np.copy(positions)
        p_best_scores = np.array([cal_score(t, p, n=num_order) for p in positions])
        g_best_position = p_best_positions[np.argmin(p_best_scores)]
        g_best_score = np.min(p_best_scores)

        MSE_per_iteration = []

        for iteration in range(iterations):
            for i in range(num_particles):
                r1, r2 = np.random.rand(2)
                p_vel = c_personal*r1*(p_best_positions[i]-positions[i])
                s_vel = c_social*r2*(g_best_position-positions[i])
                velocities[i] = inertia_weight*velocities[i] + p_vel + s_vel

                positions[i] += velocities[i]

                score = cal_score(t, positions[i], n=num_order)
                if score < p_best_scores[i]:
                    p_best_scores[i] = score
                    p_best_positions[i] = positions[i]

            best_particle_index = np.argmin(p_best_scores)
            if p_best_scores[best_particle_index] < g_best_score:
                g_best_score = p_best_scores[best_particle_index]
                g_best_position = p_best_positions[best_particle_index]

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best Score: {g_best_score}")
            MSE_per_iteration.append((g_best_score)**2)
        print(g_best_position)
        MSE_list.append((g_best_score)**2)
        MSE_sublist.append(MSE_per_iteration)
        MSE_var_list.append(np.var(MSE_per_iteration))
        j += 1
    MSE_his_list.append(MSE_sublist)
    i += 1

t_plot = np.linspace(-2*np.pi, 2*np.pi, 1000)
plt.plot(t_plot, ideal_square_wave(t_plot), label="ideal", color='blue')
plt.plot(t_plot, square_wave_fourier_series(t_plot, g_best_position, n=num_order), label="fourier approximation", color='red')
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.title("square wave approximation")
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(np.tile(lr_list,4), MSE_list, color='blue')
plt.xlabel("learning rate")
plt.ylabel("MSE")
plt.title("learning rate vs. MSE")
plt.grid()
plt.show()

plt.scatter(np.tile(lr_list,4), MSE_var_list, color='red')
plt.xlabel("learning rate")
plt.ylabel("MSE variation")
plt.title("learning rate vs. MSE variation")
plt.grid()
plt.show()

plt.scatter(np.repeat(power_list,4), MSE_list, color='green')
plt.xlabel("power")
plt.ylabel("MSE")
plt.title("power vs. MSE")
plt.grid()
plt.show()

for i in range(len(power_list)):
    for j in range(len(lr_list)):
        plt.plot(MSE_his_list[i][j], label=f"lr={lr_list[j]}, power={power_list[i]}")
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.title("epoch vs. MSE")
plt.legend(loc='upper right')
plt.grid()
plt.show()