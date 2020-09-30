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
        gradient = []
        propagation = []
        
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]    

            deriv_1 = neuron.activation_value - labels[i]
            deriv_2 = neuron.activator.derivative(neuron.sum_weights_bias_value)

            for j in range(len(neuron.input_neurons)):
                deriv_3 = self.previous_layer.neurons[j].activation_value
                deriv = deriv_1 * deriv_2 * deriv_3 
                gradient.append(deriv)

            #bias_deriv = deriv_1 * deriv_2
            propagation.append((deriv_1, deriv_2))

        back_prop = self.previous_layer.back_propagate(propagation)
        gradient.extend(back_prop)
        return gradient


    def back_propagate(self, prop):
    
        gradient = []
        propagation = []

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]

            deriv_1 = 0
            for j in range(len(prop)):
                (d1,d2) = prop[j]
                deriv_1 += d1 * d2 * self.next_layer.neurons[j].weights[i]

            deriv_2 = neuron.activator.derivative(neuron.sum_weights_bias_value)

            for k in range(len(neuron.input_neurons)):
                deriv_3 = self.previous_layer.neurons[k].activation_value
                deriv = deriv_1 * deriv_2 * deriv_3 
                gradient.append(deriv)

            propagation.append((deriv_1, deriv_2))

        #another layer left
        if self.previous_layer == None:
            return gradient
        else:
            back_prop = self.previous_layer.back_propagate(propagation)
            gradient.extend(back_prop)
            return gradient

    def descent_all(self, gradient):
        for i in range(len(self.neurons)):
            self.neurons[i].descent(gradient)
        
        if self.previous_layer != None and len(gradient) > 0:
            self.previous_layer.descent_all(gradient)