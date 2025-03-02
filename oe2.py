class Perceptron:
    def __init__(self, num_inputs=1, learning_rate=0.1):
        self.num_inputs = num_inputs
        self.weights = [0] * num_inputs
        self.bias = 0
        self.learning_rate = learning_rate
        

    def weighted_sum(self, inputs):
        weighted_sum = self.bias
        for i in range(self.num_inputs):
            weighted_sum += self.weights[i] * inputs[i]
        return weighted_sum
    
    def activation(self, weighted_sum):
        # print(f'weighted sum: {weighted_sum}')
        return 1 if weighted_sum >= 0 else -1
    
    def training(self, training_set, max_epochs=100000, learning_rate=0.001):
        for epoch in range(max_epochs):
            total_error = 0

            for inputs, actual in training_set:
                prediction = self.activation(self.weighted_sum(inputs))
                error = actual - prediction
                total_error += abs(error)

                for i in range(self.num_inputs):
                    self.weights[i] += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
            
            # print(f'Epoch {epoch + 1}, Total error: {total_error}, Weights: {self.weights}, Bias: {self.bias}')

            if total_error == 0:
                print(f'Converged after {epoch + 1} epocs')
                break

        if epoch == max_epochs:
            print('Maximum epochs reached, the training did not converge')
        
        return self.weights
    
training_data_even_odd = [
    ([-5], -1),
    ([-4], 1),
    ([-3], -1),
    ([-2], 1),
    ([-1], -1),
    ([0], 1),
    ([1], -1),
    ([2], 1),
    ([3], -1),
    ([4], 1),
    ([5], -1),
    ([6], 1),
    ([7], -1),
    ([8], 1),
    ([9], -1),
    ([10], 1),
    ([11], -1),
    ([12], 1),
    ([13], -1),
    ([14], 1),
    ([15], -1),
    ([16], 1),
    ([17], -1),
    ([18], 1),
    ([19], -1),
    ([20], 1),
    ([21], -1),
    ([22], 1),
    ([23], -1),
    ([24], 1),
    ([25], -1),
    ([26], 1), 
    ([27], -1),
    ([28], 1),
    ([29], -1),
    ([30], 1)
]

perceptron_2 = Perceptron(num_inputs=1)
perceptron_2.training(training_data_even_odd)

result_2 = perceptron_2.activation(perceptron_2.weighted_sum([1]))
result_3 = perceptron_2.activation(perceptron_2.weighted_sum([121]))

# print(result_2, result_3)

test_nums = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

for i in test_nums:
    result = perceptron_2.activation(perceptron_2.weighted_sum([i]))
    print(f'i: {i}, result: {result}')

# Output for test_nums

# i: -10, result: -1
# i: -9, result: -1
# i: -8, result: -1
# i: -7, result: -1
# i: -6, result: -1
# i: -5, result: -1
# i: -4, result: -1
# i: -3, result: -1
# i: -2, result: -1
# i: -1, result: -1
# i: 0, result: -1
# i: 1, result: 1
# i: 2, result: 1
# i: 3, result: 1
# i: 4, result: 1
# i: 5, result: 1
# i: 6, result: 1
# i: 7, result: 1
# i: 8, result: 1
# i: 9, result: 1
# i: 10, result: 1
# i: 11, result: 1
# i: 12, result: 1
# i: 13, result: 1
# i: 14, result: 1
# i: 15, result: 1
# i: 16, result: 1
# i: 17, result: 1
# i: 18, result: 1
# i: 19, result: 1
# i: 20, result: 1