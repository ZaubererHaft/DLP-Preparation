import Neuron as nn
import Activator as ac

class Layer:
    activator = None
    neurons = None
    next_layer = None
    previous_layer = None

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

        layer.previous_layer = self
        self.next_layer = layer

    def forward_propagation(self, features):
        result = []
        for i in range(len(self.neurons)):
            result.append(self.neurons[i].forward_propagation(features))

        return result

    def total_error(self, features, labels):
        sum = 0
        for i in range(len(self.neurons)):
            sum += self.neurons[i].error(features, labels[i])

        return sum     


    def gradient(self, labels):
        """
        we are layer l
        """
        result = []

        for j in range(len(self.previous_layer.neurons)):
            sum = 0

            for i in range(len(self.neurons)):
                neuron = self.neurons[i]

                deriv_1 = neuron.activation_value - labels[i]
                deriv_2 = neuron.activator.derivative(neuron.sum_weights_bias_value)
                deriv_3 = self.previous_layer.neurons[j].activation_value

                deriv = deriv_1 * deriv_2 * deriv_3

                neuron.descent_weight(deriv, i)                

            result.append(sum)

            #sum *= self.previous_layer.back_propagate(neuron.weights[j])

        return result

    def back_propagate(self, weight):
        sum = weight

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            sum *= neuron.activator.derivative(neuron.sum_weights_bias_value)

            if self.previous_layer != None:
                sum *= self.previous_layer.neurons[i].activation_value
                sum *= self.previous_layer.back_propagate(neuron.weights[i])

        return sum
