class Perceptron:
    def __init__(self, num_inputs=6, learning_rate=0.1):
        self.num_inputs = num_inputs
        self.weights = [1] * num_inputs
        self.bias = 0
        self.learning_rate = learning_rate
        

    def weighted_sum(self, inputs):
        weighted_sum = self.bias
        for i in range(self.num_inputs):
            weighted_sum += self.weights[i] * inputs[i]
            print(f'Input: {inputs[i]}, Weighted sum: {weighted_sum}, Self.Weights, {self.weights}, Bias: {self.bias}')
        return weighted_sum
    
    def activation(self, weighted_sum):
        # print(f'weighted sum: {weighted_sum}')
        return 1 if weighted_sum >= 0 else -1
    
    def training(self, training_set, max_epochs=1000, learning_rate=0.1):
        for epoch in range(max_epochs):
            total_error = 0

            for inputs, actual in training_set:
                # print(f'input: {inputs}, actual: {actual}')
                weighted_sum = self.weighted_sum(inputs)
                prediction = self.activation(weighted_sum)
                # prediction = self.activation(self.weighted_sum(inputs))
                error = actual - prediction
                total_error += abs(error)

                # print(f'input: {inputs},  prediction: {prediction}, actual: {actual}, weighted_sum: {weighted_sum}, error: {error}, total_error: {total_error}, bias: {self.bias}')

                for i in range(self.num_inputs):
                    self.weights[i] += learning_rate * error * inputs[i] 
                self.bias += learning_rate * error
            
            print(f'Epoch {epoch + 1}, Total error: {total_error}, Weights: {self.weights}, Bias: {self.bias}')

            if total_error == 0:
                print(f'Converged after {epoch + 1} epocs')
                break

        if epoch == max_epochs:
            print('Maximum epochs reached, the training did not converge')
        print(f'Return value of weights: {self.weights}')
        return self.weights
    
def create_feature_vector(num):
    even = 1 if num % 2 == 0 else 0
    positive = 1 if num > 0 else 0
    divisible_by_2 = 1 if num % 2 == 0 else 0
    divisible_by_3 = 1 if num % 3 == 0 else 0
    divisible_by_5 = 1 if num % 5 == 0 else 0
    power_of_2 = 1 if (num > 0 and (num & (num - 1)) == 0) else 0

    return [even, positive, divisible_by_2, divisible_by_3, divisible_by_5, power_of_2]

X = [create_feature_vector(i) for i in range(-5, 11)]
y = [-1 if i % 2 != 0 else 1 for i in range (-5, 11)]

perceptron = Perceptron(num_inputs=6, learning_rate=0.01)
perceptron.training(X, y)

test_inputs = [create_feature_vector(i) for i in range(-10, 11)]

print('Predictions: ')
for i, test_input in enumerate(test_inputs):
    result = perceptron.activation(perceptron.weighted_sum(test_input))
    print(f'Input: {i - 10}, Prediction: {result}')