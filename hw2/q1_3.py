import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from network import SimpleNN

def target_function(x):
    return np.sin(np.pi*x[:, 0])*np.cos(np.pi*x[:, 1])

n_train_data = 200
n_test_data = 40

# Define the 2D range
x_min, x_max = -2, 2
y_min, y_max = -2, 2

# Choose the number of points along each dimension to get close to 200 points
n_points_1d = int(np.sqrt(n_train_data))  # Close approximation for 196 points from a grid
n_test_ponits_1d = int(np.sqrt(n_test_data))

# Generate linearly spaced points for each axis
x = np.linspace(x_min, x_max, n_points_1d)
y = np.linspace(y_min, y_max, n_points_1d)
x_t = np.linspace(x_min, x_max, n_test_ponits_1d)
y_t = np.linspace(y_min, y_max, n_test_ponits_1d)

# Create a grid of x and y points
xx, yy = np.meshgrid(x, y)
xx_t, yy_t = np.meshgrid(x_t, y_t)

# Flatten the grid to get coordinate pairs
grid_points = np.column_stack((xx.ravel(), yy.ravel()))  # 196 points
t_grid_points = np.column_stack((xx_t.ravel(), yy_t.ravel()))

# Generate 4 additional random points in the same range
extra_points = np.column_stack((
    np.random.uniform(x_min, x_max, n_train_data-n_points_1d**2),
    np.random.uniform(y_min, y_max, n_train_data-n_points_1d**2)
))
t_extra_points = np.column_stack((
    np.random.uniform(x_min, x_max, n_test_data-n_test_ponits_1d**2),
    np.random.uniform(y_min, y_max, n_test_data-n_test_ponits_1d**2)
))

# Concatenate the grid points and the additional points to make 200 points
x_train = np.vstack((grid_points, extra_points))
x_test = np.vstack((t_grid_points, t_extra_points))
y_train = target_function(x_train).reshape(-1, 1)
y_train_for_plot = target_function(grid_points).reshape(xx.shape)

input_size = 2
hidden_size = 20
output_size = 1

nn = SimpleNN(
    input_size=input_size, 
    hidden_size=hidden_size, 
    output_size=output_size, 
    learning_rate=0.05, 
    seed=35,
    optimizer='adam')

epochs = 100000
vloss = []
for epoch in range(epochs):
    output = nn.forward(x_train)
    loss = nn.backward(x_train, y_train, output)
    vloss.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

y_pred = nn.forward(x_test)
y_test = target_function(x_test).reshape(-1, 1)

# Create a 3D plot of the predicted values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, y_train_for_plot, color='b', alpha=0.5, label='True Surface', rstride=100, cstride=100)

# Reshape y_pred for plotting
z_pred = y_pred.reshape(-1)

# Plot the surface or scatter points
ax.scatter(x_test[:, 0], x_test[:, 1], z_pred, c='r', marker='o')

# Set labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Predicted Z axis (f(x, y))')
ax.set_title('function 3 training result')
ax.grid()

plt.show()
plt.plot(vloss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Time")
plt.grid()
plt.show()

print(f"Network Architecture: [{input_size} {hidden_size} {output_size}]")
print(f"Activation function: [relu]")
print(f"Epochs: {epochs}, Loss: {loss:.4f}")
print(f"MSE: {np.mean((y_test-y_pred)**2):.4f}")
print(f"MSE(%) : {100*np.mean((y_test-y_pred)**2)/np.mean(y_test**2):.4f}%")