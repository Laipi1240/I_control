import numpy as np
import matplotlib.pyplot as plt
from network import SimpleNN

def target_function(x):
    return 1.0/x

# x_train = np.random.uniform(0.1, 1, 200).reshape(-1, 1)
x_train = np.linspace(0.1, 1, 200).reshape(-1, 1)
y_train = target_function(x_train)
input_size = 1
hidden_size = 10
output_size = 1

nn = SimpleNN(
    input_size=input_size, 
    hidden_size=hidden_size, 
    output_size=output_size, 
    learning_rate=0.05, 
    seed=35,
    optimizer='adam')

epochs = 10000
vloss = []
for epoch in range(epochs):
    output = nn.forward(x_train)
    loss = nn.backward(x_train, y_train, output)
    vloss.append(loss)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss:.4f}')

x_test = np.linspace(0.1, 1, 40).reshape(-1, 1)
y_test = target_function(x_test)

y_pred = nn.forward(x_test)

plt.plot(x_test, y_test, label="Original function (x^2)", color='blue')
plt.plot(x_test, y_pred, label="NN approximation", color='orange')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title("function 1 training result")
plt.grid()
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