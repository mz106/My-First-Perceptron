from perceptron_module import Perceptron

def create_perceptron_for_above_below_0():
    training_data_above_0 = {
        -10: -1, -9: -1, -8: -1, -7: -1, -6: -1,
        -5: -1, -4: -1, -3: -1, -2: -1, -1: -1,
        0: 1, 1: 1, 2: 1, 3: 1, 4: 1,
        5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1
    }

    training_data = [
        (-10, [0], -1),
        (-9, [0], -1),
        (-8, [0], -1),
        (-7, [0], -1),
        (-6, [0], -1),
        (-5, [0], -1),
        (-4, [0], -1),
        (-3, [0], -1),
        (-2, [0], -1),
        (-1, [0], -1),
        (0, [1], 1),
        (1, [1], 1),
        (2, [1], 1),
        (3, [1], 1),
        (4, [1], 1),
        (5, [1], 1),
        (6, [1], 1),
        (7, [1], 1),
        (8, [1], 1),
        (9, [1], 1),
        (10, [1], 1),
    ]

    # Convert to the required format: ([input], label)
    formatted_data = [([x], label) for x, label in training_data_above_0.items()]
    print(formatted_data)
    perceptron = Perceptron(num_inputs=1)
    perceptron.training(training_data)
    return perceptron

def create_perceptron_for_odd_even():
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

    # Convert to the required format: ([input], label)
    # formatted_data = [([x], label) for x, label in training_data_even_odd.items()]

    perceptron = Perceptron()
    perceptron.training(training_data)
    return perceptron

def create_perceptron_for_final_decision():
    training_data_final_decision = {
        (1, 1): 1,  # Above 0 and Even
        (1, -1): 2, # Above 0 and Odd
        (-1, 1): 2, # Below 0 and Even
        (-1, -1): 2  # Below 0 and Odd
    }

    # Convert to the required format: ([output_from_1, output_from_2], label)
    formatted_data = [([x[0], x[1]], label) for x, label in training_data_final_decision.items()]

    perceptron = Perceptron(num_inputs=2)
    perceptron.training(formatted_data)
    return perceptron

def classify_number(number):
    print('The Great Perceptron is starting........')
    # First perceptron: Check if number is above or below 0
    perceptron_1 = create_perceptron_for_above_below_0()
    result_1 = perceptron_1.activation(perceptron_1.weighted_sum([number]))

    # # Second perceptron: Check if number is odd or even
    # perceptron_2 = create_perceptron_for_odd_even()
    # result_2 = perceptron_2.activation(perceptron_2.weighted_sum([number]))

    # # Third perceptron: Final decision based on above/below and odd/even
    # perceptron_3 = create_perceptron_for_final_decision()
    # final_result = perceptron_3.activation(perceptron_3.weighted_sum([result_1, result_2]))

    # return f"Number {number}: Above 0? {result_1 == 1}, Even? {result_2 == 1}, Final Decision: {final_result}"

# Test the network with some numbers
# test_numbers = [-5, 0, 5, 6]
# for num in test_numbers:
#     print(classify_number(num))

classify_number(num)


