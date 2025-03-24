class Perceptron:
    def __init__(self, num_inputs=1, weights=None):
        if weights is None:
            weights = [1] * (num_inputs + 1)
        self.num_inputs = num_inputs
        self.weights = weights
        

    def weighted_sum(self, inputs):
        weighted_sum = self.weights[-1]
        for i in range(self.num_inputs):
            weighted_sum += self.weights[i] * inputs[i]
        return weighted_sum
    
    def activation(self, weighted_sum):
        return 1 if weighted_sum >= 0 else -1
    
    def training(self, training_set, max_epochs=10000, learning_rate=0.9):
        foundLine = False
        epoch = 0

        while not foundLine and epoch < max_epochs:
            epoch += 1
            total_error = 0

            for inputs, actual in training_set:
                prediction = self.activation(self.weighted_sum(inputs))
                error = actual - prediction
                total_error += abs(error)

                for i in range(self.num_inputs):
                    self.weights[i] += learning_rate * error * inputs[i]
                self.weights[-1] += learning_rate * error
            
            if total_error == 0:
                foundLine = True
        
        if epoch == max_epochs:
            print('Maximum epochs reached, the training did not converge')
        
        return self.weights
    
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
    ([20], 1)
]

perceptron_2 = Perceptron(num_inputs=1)
perceptron_2.training(training_data_even_odd)

result_2 = perceptron_2.activation(perceptron_2.weighted_sum([11]))

print(result_2)

    