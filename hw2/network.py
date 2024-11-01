import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, optimizer="adam", seed=35, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Initialize weights and biases
        np.random.seed(seed)
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Weights for input to hidden
        self.b1 = np.zeros((1, hidden_size))  # Bias for hidden layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # Weights for hidden to output
        self.b2 = np.zeros((1, output_size))  # Bias for output layer

        # Initialize Adam optimization variables if using Adam
        if optimizer == "adam":
            self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
            self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
            self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
            self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
            self.t = 0  # Initialize time step for Adam updates

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
        loss = np.mean((output - y) ** 2)  # Compute the loss (Mean Squared Error)

        # Compute gradients
        d_loss = 2 * (output - y) / m  # Gradient of loss
        dW2 = np.dot(self.a1.T, d_loss)  # Gradient w.r.t weights W2
        db2 = np.sum(d_loss, axis=0, keepdims=True)  # Gradient w.r.t bias b2

        d_a1 = np.dot(d_loss, self.W2.T)  # Gradient w.r.t a1
        d_z1 = d_a1 * self.relu_derivative(self.z1)  # Gradient w.r.t z1 (ReLU derivative)

        dW1 = np.dot(x.T, d_z1)  # Gradient w.r.t weights W1
        db1 = np.sum(d_z1, axis=0, keepdims=True)  # Gradient w.r.t bias b1

        # Update weights and biases
        if self.optimizer == "adam":
            # Increment timestep for Adam
            self.t += 1

            # Adam optimization for W1 and b1
            self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * dW1
            self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (dW1 ** 2)
            m_W1_corrected = self.m_W1 / (1 - self.beta1 ** self.t)
            v_W1_corrected = self.v_W1 / (1 - self.beta2 ** self.t)
            self.W1 -= self.learning_rate * m_W1_corrected / (np.sqrt(v_W1_corrected) + self.epsilon)

            self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1
            self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (db1 ** 2)
            m_b1_corrected = self.m_b1 / (1 - self.beta1 ** self.t)
            v_b1_corrected = self.v_b1 / (1 - self.beta2 ** self.t)
            self.b1 -= self.learning_rate * m_b1_corrected / (np.sqrt(v_b1_corrected) + self.epsilon)

            # Adam optimization for W2 and b2
            self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * dW2
            self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * (dW2 ** 2)
            m_W2_corrected = self.m_W2 / (1 - self.beta1 ** self.t)
            v_W2_corrected = self.v_W2 / (1 - self.beta2 ** self.t)
            self.W2 -= self.learning_rate * m_W2_corrected / (np.sqrt(v_W2_corrected) + self.epsilon)

            self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * db2
            self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (db2 ** 2)
            m_b2_corrected = self.m_b2 / (1 - self.beta1 ** self.t)
            v_b2_corrected = self.v_b2 / (1 - self.beta2 ** self.t)
            self.b2 -= self.learning_rate * m_b2_corrected / (np.sqrt(v_b2_corrected) + self.epsilon)

        else:  # Standard gradient descent with constant learning rate
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

        return loss
