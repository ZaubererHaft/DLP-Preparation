import Neuron as nn
import Activator as ac

class Layer:
    activator = None
    neurons = None

    def __init__(self, activator = ac.ReLu()):
        self.activator = activator
        self.neurons = []

    def add_neuron(self, name = "", initial_bias = 0):
        neuron = nn.Neuron(name, [], [1], self.activator, initial_bias)
        self.neurons.append(neuron)

    def connect_layer(self, layer, initial_weights = [[]]):

        for i in range(len(layer.neurons)):
            target_neuron = layer.neurons[i]
            target_neuron.input_neurons = self.neurons

            if len(initial_weights) > 0:
                target_neuron.weights = initial_weights[i]
            else:
                target_neuron.weights = [0] * len(self.neurons)

            assert len(target_neuron.input_neurons) == len (target_neuron.weights)

    def forward_propagation(self, features):
        result = []

        for i in range(len(self.neurons)):
            result.append(self.neurons[i].forward_propagation(features))

        return result

    def cost(self, features, labels):
        sum = 0

        for j in range(len(self.neurons)):
            neuron = self.neurons[j]

            deriv_1 = 2 * (neuron.activation(features) - labels[j])
            deriv_2 = self.activator.derivative()