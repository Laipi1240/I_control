import numpy as np
from network import SimpleNN

train_input = np.array([[0,0], [0,1], [1,0], [1,1]])
train_output = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

input_size = 2
hidden_size = 10
output_size = 1
epochs = 10000
nn = SimpleNN(
    input_size=input_size, 
    hidden_size=hidden_size, 
    output_size=output_size, 
    learning_rate=0.05, 
    seed=40,
    optimizer='adam')

epochs = 10000
vloss = []
for epoch in range(epochs):
    output = nn.forward(train_input)
    loss = nn.backward(train_input, train_output, output)
    vloss.append(loss)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss:.4f}')

print(f"Network Architecture: [{input_size} {hidden_size} {output_size}]")
print(f"Activation function: [relu]")
print(f"Epochs: {epochs}, Loss: {loss:.4f}")
print(f"input: [{train_input[0][0]} {train_input[0][1]}], Predicted: {output[0]}, Actual: {train_output[0]}")
print(f"input: [{train_input[1][0]} {train_input[1][1]}], Predicted: {output[1]}, Actual: {train_output[1]}")
print(f"input: [{train_input[2][0]} {train_input[2][1]}], Predicted: {output[2]}, Actual: {train_output[2]}")
print(f"input: [{train_input[3][0]} {train_input[3][1]}], Predicted: {output[3]}, Actual: {train_output[3]}")