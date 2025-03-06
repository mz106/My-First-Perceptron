import random
import math

class MLP:
    def __init__(self, input_size=1, hidden_size1=4, hidden_size2=4, output_size=1, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden1 = [[random.uniform(-1, 1) for _ in range(hidden_size1)] for _ in range(input_size)]
        self.bias_hidden1 = [random.uniform(-1, 1) for _ in range(hidden_size1)]
        
        self.weights_hidden1_hidden2 = [[random.uniform(-1, 1) for _ in range(hidden_size2)] for _ in range(hidden_size1)]
        self.bias_hidden2 = [random.uniform(-1, 1) for _ in range(hidden_size2)]
        
        self.weights_hidden2_output = [random.uniform(-1, 1) for _ in range(hidden_size2)]
        self.bias_output = random.uniform(-1, 1)
    
    def tanh(self, x):
        return math.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - math.tanh(x) ** 2
    
    def step_function(self, x):
        return 1 if x >= 0 else -1
    
    def forward(self, inputs):
        hidden1_activations = []
        for i in range(self.hidden_size1):
            hidden_sum = sum(self.weights_input_hidden1[j][i] * inputs[j] for j in range(self.input_size)) + self.bias_hidden1[i]
            hidden1_activations.append(self.tanh(hidden_sum))
        
        hidden2_activations = []
        for i in range(self.hidden_size2):
            hidden_sum = sum(self.weights_hidden1_hidden2[j][i] * hidden1_activations[j] for j in range(self.hidden_size1)) + self.bias_hidden2[i]
            hidden2_activations.append(self.tanh(hidden_sum))
        
        output_sum = sum(self.weights_hidden2_output[i] * hidden2_activations[i] for i in range(self.hidden_size2)) + self.bias_output
        output = self.step_function(output_sum)
        return hidden1_activations, hidden2_activations, output
    
    def train(self, training_data, epochs=100000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, expected in training_data:
                hidden1_activations, hidden2_activations, output = self.forward(inputs)
                error = expected - output
                total_error += abs(error)
                
                # Backpropagation (output to hidden2 layer)
                for i in range(self.hidden_size2):
                    self.weights_hidden2_output[i] += self.learning_rate * error * hidden2_activations[i]
                self.bias_output += self.learning_rate * error
                
                # Backpropagation (hidden2 to hidden1 layer)
                hidden2_errors = [error * self.weights_hidden2_output[i] * self.tanh_derivative(hidden2_activations[i]) for i in range(self.hidden_size2)]
                for i in range(self.hidden_size2):
                    for j in range(self.hidden_size1):
                        self.weights_hidden1_hidden2[j][i] += self.learning_rate * hidden2_errors[i] * hidden1_activations[j]
                    self.bias_hidden2[i] += self.learning_rate * hidden2_errors[i]
                
                # Backpropagation (hidden1 to input layer)
                hidden1_errors = [sum(hidden2_errors[j] * self.weights_hidden1_hidden2[i][j] for j in range(self.hidden_size2)) * self.tanh_derivative(hidden1_activations[i]) for i in range(self.hidden_size1)]
                for i in range(self.hidden_size1):
                    for j in range(self.input_size):
                        self.weights_input_hidden1[j][i] += self.learning_rate * hidden1_errors[i] * inputs[j]
                    self.bias_hidden1[i] += self.learning_rate * hidden1_errors[i]
                
            if total_error == 0:
                print(f'Converged after {epoch + 1} epochs')
                break
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Total Error: {total_error}')

# Training data for even/odd classification
training_data = [([x], 1 if x % 2 == 0 else -1) for x in range(1, 31)]

# Create and train the MLP model
mlp = MLP(input_size=1, hidden_size1=4, hidden_size2=4, output_size=1, learning_rate=0.01)
mlp.train(training_data)

# Testing
test_values = list(range(1, 21))
for value in test_values:
    _, _, prediction = mlp.forward([value])
    print(f'Input: {value}, Prediction: {prediction}')