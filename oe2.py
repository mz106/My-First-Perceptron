class Perceptron:
    def __init__(self, num_inputs=1, learning_rate=0.1):
        self.num_inputs = num_inputs
        # self.weights = [0] * num_inputs original
        # self.weights = [6.1999999999947235] 1st 
        #self.weights = [3.9999999999893956] #2nd
        # self.bias = -31.79999999999992 1st
        #self.bias = -31.79999999999992 #
        self.weights = [1]
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
    
    def training(self, training_set, max_epochs=100000, learning_rate=0.1):
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
    
# ([-5], -1),
#     ([-4], 1),
#     ([-3], -1),
#     ([-2], 1),
#     ([-1], -1),
#     ([0], 1),
    
training_data_even_odd = [
    
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

# training_data_even_odd = [
#     ([0], 1),
#     ([2], 1), 
#     ([4], 1),
#     ([6], 1),
#     ([8], 1),
#     ([10], 1),
#     ([12], 1),
#     ([14], 1),
#     ([16], 1),
#     ([18], 1),
#     ([20], 1),
#     ([1], -1),
#     ([3], -1),
#     ([5], -1),
#     ([7], -1),
#     ([9], -1),
#     ([11], -1),
#     ([13], -1),
#     ([15], -1),
#     ([17], -1),
#     ([19], -1),
# ]

perceptron_2 = Perceptron(num_inputs=1, learning_rate=0.0001)
perceptron_2.training(training_data_even_odd)

result_2 = perceptron_2.activation(perceptron_2.weighted_sum([1]))
result_3 = perceptron_2.activation(perceptron_2.weighted_sum([121]))

print('Results 2 and 3', result_2, result_3)
# -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 

test_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

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