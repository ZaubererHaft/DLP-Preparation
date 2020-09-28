import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(z):
    return max(0,z)

def id(x):
    return x

class Neuron:
    activation_function = None
    input_neurons = None   
    name = None
    weights = None
    bias = None

    def __init__(self, name, input_neurons, weights, activation_function = relu, bias = 0):
        self.name = name
        self.activation_function = activation_function
        self.input_neurons = input_neurons
        self.weights = weights
        self.bias = 0

    def predict(self, features):
        assert len(self.weights) == len(features)

        sum = 0
        for i in range(len(self.weights)):
            sum += features[i] * self.weights[i]
        sum += self.bias
        
        return self.activation_function(sum)

    def forward_propagation(self, features):
        inputs = []

        for i in range(len(self.input_neurons)):
            input_neuron = self.input_neurons[i]
            inputs.append(input_neuron.forward_propagation(features))

        if len(inputs) <= 0:
            return self.predict(features)

        return self.predict(inputs)