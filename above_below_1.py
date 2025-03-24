
# 1. If multiplied by -1 is the result negative? yes - even, no - odd
# 2. if x + [x] != 0 - negative, = 0 - positive
# 3. if x/x = -1 = - negative, = 1 - positive 

from perceptron_module import Perceptron

training_data = [
    (-5, [0, 1, 1], -1),
    (-4, [0, 1, 1], -1),
    (-3, [0, 1, 1], -1),
    (-2, [0, 1, 1], -1),
    (-1, [0, 1, 1], -1),
    (0, [0, 0, 0 ], 1),
    (1, [1, 0, 0 ], 1),
    (2, [1, 0, 0], 1),
    (3, [1, 0, 0], 1),
    (4, [1, 0, 0], 1),
    (5, [1, 0, 0], 1),
]

perceptron = Perceptron(num_inputs=3)
weights = perceptron.training(training_data)
print(f'Weights post training: {weights}')

test_nums = [
    (-50, [0, 1, 1]),
    (6, [1, 0, 0]),
    (192831, [1, 0, 0]),
    (-84928123721, [0, 1, 1]), 
    (-292, [0, 1, 1]),
    (0, [0, 0, 0])
]

for num, inputs in test_nums:
    prediction = perceptron.activation(perceptron.weighted_sum(inputs))
    print(f'Result for {num}: {prediction}')


