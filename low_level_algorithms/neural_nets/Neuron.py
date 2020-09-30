import math
import Activator as ac

def create_input_neuron(name):
    """
    Creates a special neuron, the input neuron.
    It has no bias, no other inputs, weights of 1 and the identity function as __activation function

    Returns:
            Neuron that serves as input neuron
    """
    return Neuron(name, [], [1], ac.Identity())

class Neuron:
    activator = None
    input_neurons = None   
    name = None
    weights = None
    bias = None
    learning_rate = None

    activation_value = None
    sum_weights_bias_value = None
    

    def __init__(self, name, input_neurons, weights, activatior = ac.ReLu, bias = 0, learning_rate = 0.01):
        self.name = name
        self.activator = activatior
        self.input_neurons = input_neurons
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

        self.activation_value = 0
        self.sum_weights_bias_value = 0

    #activation value a
    def __activation(self, features):
        assert len(self.weights) == len(features)

        sum = self.__sum_weights_bias(features)
        activation =  self.activator.function(sum)

        self.activation_value = activation
        return activation

    #sum of weights and biases z
    def __sum_weights_bias(self, features):
        sum = 0
        for i in range(len(self.weights)):
            sum += features[i] * self.weights[i]
        sum += self.bias

        self.sum_weights_bias_value = sum
        return sum

    def forward_propagation(self, features):
        inputs = []

        for i in range(len(self.input_neurons)):
            input_neuron = self.input_neurons[i]
            prop = None

            #propagate input as long until the predecessor is an input feature
            if len(input_neuron.input_neurons) <= 0:
                prop = input_neuron.forward_propagation([features[i]])
            else:
                prop = input_neuron.forward_propagation(features)

            inputs.append(prop)
            
        if len(inputs) <= 0:
            return self.__activation(features)

        return self.__activation(inputs)

    def descent_weight(self, weight, weight_index):
        self.weights[weight_index] -= self.learning_rate * weight

    def descent_bias(self, bias): 
        self.bias -= self.learning_rate * bias

    def error(self, features, labeled_output):
        prediction = self.forward_propagation(features)
        return 0.5 * (prediction - labeled_output)**2
