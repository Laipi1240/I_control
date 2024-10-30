import numpy as np
import matplotlib.pyplot as plt
import SimpleNN

def target_function(x):
    return 1.0/x

# x_train = np.random.uniform(0.1, 1, 200).reshape(-1, 1)
x_train = np.linspace(0.1, 1, 200).reshape(-1, 1)
y_train = target_function(x_train)

nn = SimpleNN(input_size=1, hidden_size=10, output_size=1, learning_rate=0.05)

epochs = 10000

for epoch in range(epochs):
    output = nn.forward(x_train)
    loss = nn.backward(x_train, y_train, output)

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
plt.grid()
plt.show()