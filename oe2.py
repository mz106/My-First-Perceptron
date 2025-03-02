class Perceptron:
    def __init__(self, num_inputs=1, learning_rate=0.1):
        self.num_inputs = num_inputs
        self.weights = [0] * num_inputs
        self.bias = 0
        self.learning_rate = learning_rate
        

    def weighted_sum(self, inputs):
        weighted_sum = self.weights[-1]
        for i in range(self.num_inputs):
            weighted_sum += self.weights[i] * inputs[i]
        return weighted_sum
    
    def activation(self, weighted_sum):
        # print(f'weighted sum: {weighted_sum}')
        return 1 if weighted_sum >= 0 else -1
    
    def training(self, training_set, max_epochs=10000, learning_rate=0.01):
        for epoch in range(max_epochs):
            total_error = 0

            for inputs, actual in training_set:
                prediction = self.activation(self.weighted_sum(inputs))
                error = actual - prediction
                total_error += abs(error)

                for i in range(self.num_inputs):
                    self.weights[i] += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
            
            print(f'Epoch {epoch + 1}, Total error: {total_error}, Weights: {self.weights}, Bias: {self.bias}')

            if total_error == 0:
                print(f'Converged after {epoch + 1} epocs')
                break

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
result_3 = perceptron_2.activation(perceptron_2.weighted_sum([2]))

print(result_2, result_3)