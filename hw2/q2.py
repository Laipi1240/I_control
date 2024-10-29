import numpy as np

train_input = np.array([[0,0], [0,1], [1,0], [1,1]])
train_output = np.array([[0], [1], [1], [0]])

n_input = 2
n_hidden = 4
n_output = 1

w1 = np.random.rand(n_input, n_hidden)
b1 = np.random.rand(1, n_hidden)
w2 = np.random.rand(n_hidden, n_output)
b2 = np.random.rand(1, n_output)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    hidden_input = np.dot(train_input, w1) + b1
    hidden_output = sigmoid(hidden_input)
    outlayer_input = np.dot(hidden_output, w2) + b2
    outlayer_output = sigmoid(outlayer_input)

    error = train_output - outlayer_output

    d_output = error * sigmoid_derivative(outlayer_output)
    error_hidden = d_output.dot(w2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    w2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    w1 += train_input.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Error: {np.mean(np.abs(error))}")

print("\nOutput after training:")
print(outlayer_output)