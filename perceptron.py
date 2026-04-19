import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=20):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        return 1 if x > 0 else 0

    def train(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        print(self.weights)
        self.bias = 0

        print(f"Training with Learning Rate = {self.lr}\n")

        for epoch in range(self.epochs):
            total_error = 0
            
            for i in range(len(X)):
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.activation(linear_output)
                
                error = y[i] - prediction
                total_error += abs(error)
                
                self.weights += self.lr * error * X[i]
                print(f"learning weight= {self.weights}")      
                self.bias += self.lr * error    
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:2d} | Error: {total_error} | Weights: {self.weights} | Bias: {self.bias:.3f}")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

perceptron = Perceptron(learning_rate=0.5, epochs=15)

perceptron.train(X, y)

print("\nFinal Predictions:")
for i in range(len(X)):
    pred = perceptron.activation(np.dot(X[i], perceptron.weights) + perceptron.bias)
    print(f"{X[i]} → Predicted: {pred} (Actual: {y[i]})")