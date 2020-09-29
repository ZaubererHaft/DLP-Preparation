import math
import Activator as ac

def create_input_neuron(name):
    """
    Creates a special neuron, the input neuron.
    It has no bias, no other inputs, weights of 1 and the identity function as activation function

    Returns:
            Neuron that serves as input neuron
    """
    return Neuron(name, [], [1], ac.Identity())

def error(prediction, labeled_output):
    """
    Error function that we tweak
    """
    return 0.5 * (prediction - labeled_output)**2

class Neuron:
    #activator sigma
    activator = None
    input_neurons = None   
    name = None
    weights = None
    bias = None

    def __init__(self, name, input_neurons, weights, activatior = ac.ReLu, bias = 0):
        self.name = name
        self.activator = activatior
        self.input_neurons = input_neurons
        self.weights = weights
        self.bias = bias

    #activation value a
    def activation(self, features):
        assert len(self.weights) == len(features)
        sum = self.sum_weights_bias(features)
        return self.activator.function(sum)

    #sum of weights and biases z
    def sum_weights_bias(self, features):
        sum = 0
        for i in range(len(self.weights)):
            sum += features[i] * self.weights[i]
        sum += self.bias
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
            return self.activation(features)

        return self.activation(inputs)
