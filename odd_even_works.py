

# if x % 2 == 0:
#     return 1
# else:
#     return -1

# is is divisible by 2
# is divisible by 3
# is divisible by 5
# is a perfect square

from perceptron_module import Perceptron 

# class Perceptron:
#     def __init__(self, num_inputs=4, num_hidden=2, num_outputs=1, learning_rate=0.01):
#         self.num_inputs = num_inputs
#         self.num_hidden = num_hidden
#         self.num_outputs = num_outputs
#         self.learning_rate = learning_rate
#         self.bias = 0
#         # self.weights = [1, 0.3, 0.5, 0.5]
#         self.weights = [0, 0, 0, 0]


#     def weighted_sum(self, inputs):
#         weighted_sum = self.bias
#         for i in range(self.num_inputs):
#             weighted_sum += self.weights[i] * inputs[i]
#         #     print(f'Input: {inputs[i]}, Weighted sum: {weighted_sum}, Self.Weights, {self.weights}, Bias: {self.bias}')
#         return weighted_sum
    
#     def activation(self, weighted_sum):
#         # print(f'weighted sum: {weighted_sum}')
#         return 1 if weighted_sum >= 0 else -1
    
#     def training(self, training_set, max_epochs=10000, learning_rate=0.1):
#         for epoch in range(max_epochs):
#             total_error = 0

#             for num, inputs, actual in training_set:
#                 # print(f'num: {num}, input: {inputs}, actual: {actual}')
#                 weighted_sum = self.weighted_sum(inputs)
#                 prediction = self.activation(weighted_sum)
#                 prediction = self.activation(self.weighted_sum(inputs))
#                 error = actual - prediction
#                 total_error += abs(error)

#                 print(f'input: {inputs},  prediction: {prediction}, actual: {actual}, weighted_sum: {weighted_sum}, error: {error}, total_error: {total_error}, bias: {self.bias}')

#                 for i in range(self.num_inputs):
#                     self.weights[i] += learning_rate * error * inputs[i] 
#                 self.bias += learning_rate * error
            
#             print(f'Epoch {epoch + 1}, Total error: {total_error}, Weights: {self.weights}, Bias: {self.bias}')

#             if total_error == 0:
#                 print(f'Converged after {epoch + 1} epocs')
#                 break

#         if epoch == max_epochs:
#             print('Maximum epochs reached, the training did not converge')
#         print(f'Return value of weights: {self.weights}')
#         return self.weights




training_data = [
    (0, [1,1,1,1], 1),
    (1, [0,0,0,0], -1),
    (2, [1,0,0,0], 1),
    (3, [0,1,0,0], -1),
    (4, [1,0,0,1], 1),
    (5, [0,0,1,0], -1),
    (6, [1,1,0,0], 1),
    (7, [0,0,0,0], -1),
    (8, [1,0,0,0], 1),
    (9, [0,1,0,1], -1),
    (10, [1,0,0,0], 1)
]

perceptron = Perceptron()
weights = perceptron.training(training_data)

# test_nums = [
#     (1, [0,0,0,0]),
#     (2, [1,0,0,0]),
#     (3, [0,1,0,0]),
#     (4, [1,0,0,1]),
#     (5, [0,0,1,0]),
#     (11, [0,0,0,0]),
#     (56, [2, ])
# ]

# test_nums = [
#     (-10, [1, 0, 1, 0]),  # Divisible by 2, 5
#     (-9, [0, 1, 0, 0]),   # Divisible by 3
#     (-8, [1, 0, 0, 0]),   # Divisible by 2
#     (-7, [0, 0, 0, 0]),   # Not divisible by 2, 3, 5, not a perfect square
#     (-6, [1, 1, 0, 0]),   # Divisible by 2, 3
#     (-5, [0, 0, 1, 0]),   # Divisible by 5
#     (-4, [1, 0, 0, 1]),   # Divisible by 2, perfect square (-2)^2 = 4
#     (-3, [0, 1, 0, 0]),   # Divisible by 3
#     (-2, [1, 0, 0, 0]),   # Divisible by 2
#     (-1, [0, 0, 0, 0]),   # Not divisible by 2, 3, 5, not a perfect square
#     (0, [1, 1, 1, 1]),    # Special case: Divisible by everything, perfect square
#     (1, [0, 0, 0, 0]),    # Not divisible by 2, 3, 5, not a perfect square
#     (2, [1, 0, 0, 0]),    # Divisible by 2
#     (3, [0, 1, 0, 0]),    # Divisible by 3
#     (4, [1, 0, 0, 1]),    # Divisible by 2, perfect square
#     (5, [0, 0, 1, 0]),    # Divisible by 5
#     (6, [1, 1, 0, 0]),    # Divisible by 2, 3
#     (9, [0, 1, 0, 1]),    # Divisible by 3, perfect square
#     (10, [1, 0, 1, 0]),   # Divisible by 2, 5
# ]

test_nums = [
    (-547, [0, 0, 0, 0]),  # Not divisible by 2, 3, or 5, not a perfect square
    (74, [1, 0, 0, 0]),    # Divisible by 2
    (19987, [0, 0, 0, 0]), # Not divisible by 2, 3, or 5, not a perfect square
    (-647, [0, 0, 0, 0]),  # Not divisible by 2, 3, or 5, not a perfect square
    (84, [1, 1, 0, 0])     # Divisible by 2 and 3
]


for num, inputs in test_nums:
    prediction = perceptron.activation(perceptron.weighted_sum(inputs))
    # print(f'Number: {num}, Inputs: {inputs}, Prediction: {prediction}')

# for i in test_nums:
#     result = perceptron.activation(perceptron.weighted_sum(weights))
#     print(f'i: {i}, result: {result}')

# 37 [0, 0, 0, 0]
# 100 [1, 0, 1, 1]

result_37 = prediction = perceptron.activation(perceptron.weighted_sum([0, 0, 0, 0]))
result_100 = prediction = perceptron.activation(perceptron.weighted_sum([1, 0, 1, 1]))

result_55 = prediction = perceptron.activation(perceptron.weighted_sum([0, 0, 1, 0]))
result_neg_7839 = prediction = perceptron.activation(perceptron.weighted_sum([0, 0, 1, 0]))
result_neg_78392 = prediction = perceptron.activation(perceptron.weighted_sum([1, 0, 0, 0]))
print(f'Result 37: {result_37}, Result 100: {result_100}, Result 55: {result_55}, Result -7839: {result_neg_7839}, Result -78392: {result_neg_78392}')








