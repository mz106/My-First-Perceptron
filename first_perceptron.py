# Based on Codecademy course 

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

    def training(self, training_set):
        foundLine = False
        while not foundLine:
            total_error = 0
            for inputs, actual in training_set:
                prediction = self.activation(self.weighted_sum(inputs))
                error = actual - prediction
                total_error += abs(error)
                for i in range(self.num_inputs):
                    self.weights[i] += error * inputs[i]  
            if total_error == 0:
                foundLine = True

def create_perceptron_for_above_below_0():
    training_data_above_0 = {
        -10: -1, -9: -1, -8: -1, -7: -1, -6: -1,
        -5: -1, -4: -1, -3: -1, -2: -1, -1: -1,
        0: 1, 1: 1, 2: 1, 3: 1, 4: 1,
        5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1
    }

    # Convert to the required format: ([input], label)
    formatted_data = [([x], label) for x, label in training_data_above_0.items()]

    perceptron = Perceptron(num_inputs=1)
    perceptron.training(formatted_data)
    return perceptron

# Perceptron for Odd/Even check
def create_perceptron_for_odd_even():
    training_data_even_odd = {
        1: -1, 2: 1, 3: -1, 4: 1, 5: -1, 6: 1, 7: -1, 8: 1,
        9: -1, 10: 1
    }

    # Convert to the required format: ([input], label)
    formatted_data = [([x], label) for x, label in training_data_even_odd.items()]

    perceptron = Perceptron(num_inputs=1)
    perceptron.training(formatted_data)
    return perceptron

# Perceptron for final decision
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

# Combine the two perceptrons and make a final decision
def classify_number(number):
    print('The Great Perceptron is starting........')
    # First perceptron: Check if number is above or below 0
    perceptron_1 = create_perceptron_for_above_below_0()
    result_1 = perceptron_1.activation(perceptron_1.weighted_sum([number]))

    # Second perceptron: Check if number is odd or even
    perceptron_2 = create_perceptron_for_odd_even()
    result_2 = perceptron_2.activation(perceptron_2.weighted_sum([number]))

    # Third perceptron: Final decision based on above/below and odd/even
    perceptron_3 = create_perceptron_for_final_decision()
    final_result = perceptron_3.activation(perceptron_3.weighted_sum([result_1, result_2]))

    return f"Number {number}: Above 0? {result_1 == 1}, Even? {result_2 == 1}, Final Decision: {final_result}"

# Test the network with some numbers
test_numbers = [-5, 0, 5, 6]
for num in test_numbers:
    print(classify_number(num))