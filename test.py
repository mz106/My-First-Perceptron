class Perceptron:
    def __init__(self, num_inputs=1, weights=[1]):
        self.num_inputs = num_inputs
        self.weights = weights  

    def weighted_sum(self, inputs):
        weighted_sum = 0
        for i in range(self.num_inputs):
            weighted_sum += self.weights[i] * inputs[i]  
        return weighted_sum

    def activation(self, weighted_sum):
        return 1 if weighted_sum >= 0 else -1

    def training(self, training_set, max_epochs=1000):
        foundLine = False
        epoch = 0
        while not foundLine and epoch < max_epochs:
            epoch += 1
            total_error = 0
            print(total_error)
            for inputs, actual in training_set:
                prediction = self.activation(self.weighted_sum(inputs))
                error = actual - prediction
                total_error += abs(error)
                for i in range(self.num_inputs):
                    self.weights[i] += error * inputs[i]  
            if total_error == 0:
                foundLine = True

        if epoch == max_epochs:
            print('Max epochs reached, training did not converge')

training_data_above_0 = {
        -10: -1, -9: -1, -8: -1, -7: -1, -6: -1,
        -5: -1, -4: -1, -3: -1, -2: -1, -1: -1,
        0: 1, 1: 1, 2: 1, 3: 1, 4: 1,
        5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1
    }

formatted_data_1 = [([x], label) for x, label in training_data_above_0.items()]

# perceptron_1 = Perceptron(num_inputs=1)
# perceptron_1.training(formatted_data_1)

# result_1 = perceptron_1.activation(perceptron_1.weighted_sum([18]))

# print(result_1)

#====================================

# training_data_even_odd = {
#         1: -1, 2: 1, 3: -1, 4: 1, 5: -1, 6: 1, 7: -1, 8: 1,
#         9: -1, 10: 1
#     }

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
    ([10], 1)
]

# formatted_data_2 = [([x], label) for x, label in training_data_even_odd.items()]

perceptron_2 = Perceptron(num_inputs=1)
perceptron_2.training(training_data_even_odd)

result_2 = perceptron_2.activation(perceptron_2.weighted_sum([18]))

print(result_2)