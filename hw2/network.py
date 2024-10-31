import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, seed=35):
        # Initialize weights and biases
        np.random.seed(seed)
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Weights for input to hidden
        self.b1 = np.zeros((1, hidden_size))  # Bias for hidden layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # Weights for hidden to output
        self.b2 = np.zeros((1, output_size))  # Bias for output layer

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        # Forward pass
        self.z1 = np.dot(x, self.W1) + self.b1  # Input to hidden layer
        self.a1 = self.relu(self.z1)  # Activation of hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Hidden to output layer
        return self.z2  # Output layer (raw scores)

    def backward(self, x, y, output):
        # Backward pass (backpropagation)
        m = y.shape[0]  # Number of samples
        # Compute the loss (Mean Squared Error)
        loss = np.mean((output - y) ** 2)

        # Compute gradients
        d_loss = 2 * (output - y) / m  # Gradient of loss
        dW2 = np.dot(self.a1.T, d_loss)  # Gradient w.r.t weights W2
        db2 = np.sum(d_loss, axis=0, keepdims=True)  # Gradient w.r.t bias b2

        d_a1 = np.dot(d_loss, self.W2.T)  # Gradient w.r.t a1
        d_z1 = d_a1 * self.relu_derivative(self.z1)  # Gradient w.r.t z1 (ReLU derivative)

        dW1 = np.dot(x.T, d_z1)  # Gradient w.r.t weights W1
        db1 = np.sum(d_z1, axis=0, keepdims=True)  # Gradient w.r.t bias b1

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        return loss