import numpy as np

# =========================
# Activation Functions
# =========================

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative)
}

# =========================
# Autoencoder Class
# =========================

class Autoencoder:
    def __init__(
        self,
        layer_sizes,
        activations,
        learning_rate=0.01,
        l2_lambda=0.001
    ):
        """
        layer_sizes: list (e.g. [784, 256, 128, 32, 128, 256, 784])
        activations: list of activation names per layer (excluding input)
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.num_layers = len(layer_sizes) - 1

        self._init_weights()

    def _init_weights(self):
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            w = np.random.randn(
                self.layer_sizes[i],
                self.layer_sizes[i + 1]
            ) * np.sqrt(2 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i + 1]))

            self.weights.append(w)
            self.biases.append(b)

    # =========================
    # Forward Pass
    # =========================
    def forward(self, X):
        activations = [X]
        zs = []

        for i in range(self.num_layers):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            zs.append(z)

            act_func, _ = ACTIVATIONS[self.activations[i]]
            a = act_func(z)
            activations.append(a)

        return activations, zs

    # =========================
    # Backpropagation
    # =========================
    def backward(self, X, activations, zs):
        grads_w = []
        grads_b = []

        # MSE derivative
        delta = (activations[-1] - X)

        for i in reversed(range(self.num_layers)):
            _, act_deriv = ACTIVATIONS[self.activations[i]]
            delta *= act_deriv(zs[i])

            dw = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            # L2 regularization
            dw += self.l2_lambda * self.weights[i]

            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            delta = delta @ self.weights[i].T

        return grads_w, grads_b

    # =========================
    # Update Parameters
    # =========================
    def update(self, grads_w, grads_b):
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    # =========================
    # Training
    # =========================
    def train(self,X,epochs=50,batch_size=32,lr_decay=0.95, decay_step=10):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]

            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i:i + batch_size]

                activations, zs = self.forward(batch)
                grads_w, grads_b = self.backward(batch, activations, zs)
                self.update(grads_w, grads_b)

            # Learning rate scheduling
            if (epoch + 1) % decay_step == 0:
                self.learning_rate *= lr_decay

            loss = np.mean((self.forward(X)[0][-1] - X) ** 2)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}, LR: {self.learning_rate:.6f}")

    # =========================
    # Encode / Decode
    # =========================
    def encode(self, X):
        activations, _ = self.forward(X)
        bottleneck_index = len(self.layer_sizes) // 2
        return activations[bottleneck_index]

    def reconstruct(self, X):
        return self.forward(X)[0][-1]


# =========================
# Example Usage
# =========================

if __name__ == "__main__":
    # Dummy data
    X = np.random.rand(1000, 20)

    # Autoencoder architecture
    layers = [20, 64, 32, 8, 32, 64, 20]
    activations = ["relu", "relu", "tanh", "relu", "relu", "sigmoid"]

    ae = Autoencoder( 
        layer_sizes=layers,
        activations=activations,
        learning_rate=0.01,
        l2_lambda=0.001
    )

    ae.train(
        X,
        epochs=50,
        batch_size=32,
        lr_decay=0.9,
        decay_step=10
    )
