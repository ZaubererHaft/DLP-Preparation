import math

def sigmoid(x):
    """
    Sigmoid function returning a value between 0 and 1
    """
    return 1 / (1 + math.exp(-x))

def relu(z):
    """
    Rectified Linear Unit
    """
    return max(0,z)

def id(x):
    """
    Identity function returning the identity of an object
    """
    return x

def create_input_neuron(name):
    """
    Creates a special neuron, the input neuron.
    It has no bias, no other inputs, weights of 1 and the identity function as activation function

    Returns:
            Neuron that serves as input neuron
    """
    return Neuron(name, [], [1], id)

def error(prediction, labeled_output):
    """
    Error function that we tweak
    """
    return 0.5 * (prediction - labeled_output)**2

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
        self.bias = bias

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
            prop = None

            #propagate input as long until the predecessor is an input feature
            if len(input_neuron.input_neurons) <= 0:
                prop = input_neuron.forward_propagation([features[i]])
            else:
                prop = input_neuron.forward_propagation(features)

            inputs.append(prop)
            
        if len(inputs) <= 0:
            return self.predict(features)

        return self.predict(inputs)

    def current_error(self, features):
        return 0


