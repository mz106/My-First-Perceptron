import random
import math

class MLP:
    def __init__(self, input_size=1, hidden_size=2, output_size=1, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        
        self.weights_hidden_output = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_output = random.uniform(-1, 1)
    
    def relu(self, x):
        return max(0, x)
    
    def relu_derivative(self, x):
        return 1 if x > 0 else 0
    
    def step_function(self, x):
        return 1 if x >= 0 else -1
    
    def forward(self, inputs):
        hidden_activations = []
        for i in range(self.hidden_size):
            hidden_sum = sum(self.weights_input_hidden[j][i] * inputs[j] for j in range(self.input_size)) + self.bias_hidden[i]
            hidden_activations.append(self.relu(hidden_sum))
        
        output_sum = sum(self.weights_hidden_output[i] * hidden_activations[i] for i in range(self.hidden_size)) + self.bias_output
        output = self.step_function(output_sum)
        return hidden_activations, output
    
    def train(self, training_data, epochs=100000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, expected in training_data:
                hidden_activations, output = self.forward(inputs)
                error = expected - output
                total_error += abs(error)
                
                # Backpropagation (output to hidden layer)
                for i in range(self.hidden_size):
                    self.weights_hidden_output[i] += self.learning_rate * error * hidden_activations[i]
                self.bias_output += self.learning_rate * error
                
                # Backpropagation (hidden to input layer)
                for i in range(self.hidden_size):
                    hidden_error = error * self.weights_hidden_output[i] * self.relu_derivative(hidden_activations[i])
                    for j in range(self.input_size):
                        self.weights_input_hidden[j][i] += self.learning_rate * hidden_error * inputs[j]
                    self.bias_hidden[i] += self.learning_rate * hidden_error
                
            if total_error == 0:
                print(f'Converged after {epoch + 1} epochs')
                break
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Total Error: {total_error}')

# Training data for even/odd classification
training_data = [([x], 1 if x % 2 == 0 else -1) for x in range(1, 31)]

# Create and train the MLP model
mlp = MLP(input_size=1, hidden_size=3, output_size=1, learning_rate=0.01)
mlp.train(training_data)

# Testing
test_values = list(range(1, 21))
for value in test_values:
    _, prediction = mlp.forward([value])
    print(f'Input: {value}, Prediction: {prediction}')