class Perceptron:
    def __init__(self, num_inputs=4, num_hidden=2, num_outputs=1, learning_rate=0.01):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.bias = 0
        # self.weights = [1, 0.3, 0.5, 0.5]
        self.weights = [0, 0, 0, 0]


    def weighted_sum(self, inputs):
        weighted_sum = self.bias
        for i in range(self.num_inputs):
            weighted_sum += self.weights[i] * inputs[i]
        #     print(f'Input: {inputs[i]}, Weighted sum: {weighted_sum}, Self.Weights, {self.weights}, Bias: {self.bias}')
        return weighted_sum
    
    def activation(self, weighted_sum):
        # print(f'weighted sum: {weighted_sum}')
        return 1 if weighted_sum >= 0 else -1
    
    def training(self, training_set, max_epochs=10000, learning_rate=0.1):
        for epoch in range(max_epochs):
            total_error = 0

            for num, inputs, actual in training_set:
                # print(f'num: {num}, input: {inputs}, actual: {actual}')
                weighted_sum = self.weighted_sum(inputs)
                prediction = self.activation(weighted_sum)
                prediction = self.activation(self.weighted_sum(inputs))
                error = actual - prediction
                total_error += abs(error)

                print(f'input: {inputs},  prediction: {prediction}, actual: {actual}, weighted_sum: {weighted_sum}, error: {error}, total_error: {total_error}, bias: {self.bias}')

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