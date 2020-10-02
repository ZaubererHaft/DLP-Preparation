import math
import Neuron as nn
import Layer as la
import Activator as ac

"""
i1 = nn.create_input_neuron("i1")
i2 = nn.create_input_neuron("i2")

h1 = nn.Neuron("h1",[i1, i2],[0.15, 0.2], ac.Sigmoid(), 0.35)
h2 = nn.Neuron("h2",[i1, i2],[0.25, 0.3], ac.Sigmoid(), 0.35)

o1 = nn.Neuron("o1",[h1, h2],[0.40, 0.45], ac.Sigmoid(), 0.60)
o2 = nn.Neuron("o2",[h1, h2],[0.50, 0.55], ac.Sigmoid(), 0.60)

print(h1.forward_propagation([0.05, 0.10]))
print(h2.forward_propagation([0.05, 0.10]))

out_o1 = o1.forward_propagation([0.05, 0.10])
out_o2 = o2.forward_propagation([0.05, 0.10])

print(out_o1)
print(out_o2)

print(o1.error([0.05, 0.10], 0.01))
print(o2.error([0.05, 0.10], 0.99))

#----------------------------------------------------here with layers---------------------------------
input_layer = la.Layer(ac.Identity())
input_layer.add_neuron("i1")
input_layer.add_neuron("i2")

hidden_layer_1 = la.Layer(ac.Sigmoid())
hidden_layer_1.add_neuron("h1", 0.35)
hidden_layer_1.add_neuron("h2", 0.35)

output_layer = la.Layer(ac.Sigmoid())
output_layer.add_neuron("o1", 0.6)
output_layer.add_neuron("o2", 0.6)

input_layer.connect_layer(hidden_layer_1, [[0.15, 0.2],[0.25, 0.3]])
hidden_layer_1.connect_layer(output_layer, [[0.40, 0.45],[0.50, 0.55]])

print(output_layer.forward_propagation([0.05, 0.10]))
print(output_layer.total_error([0.05, 0.10],[0.01,0.99]))
"""

#----------------------------------------------------------------------------------------------------
input = la.Layer(ac.Identity())
hidden = la.Layer(ac.Sigmoid())
output = la.Layer(ac.Sigmoid())

input.add_neuron("x1")
input.add_neuron("x2")
input.add_neuron("x3")

hidden.add_neuron("h1", 0.5)
hidden.add_neuron("h2", 0.5)

output.add_neuron("o1", 0.5)
output.add_neuron("o2", 0.5)

input.connect_layer(hidden, [[0.1,0.3,0.5], [0.2,0.4,0.6]])
hidden.connect_layer(output, [[0.7,0.9], [0.8,0.1]])

print("-------------------forward propagation values---------------------------")
output.forward_propagation([1,4,5])

print(output.neurons[0].activation_value)
print(output.neurons[1].activation_value)

print(hidden.neurons[0].activation_value)
print(hidden.neurons[1].activation_value)


print("-------------------backward propagation values---------------------------")
grad = output.gradient([0.1,0.05])
print(grad)
output.descent_all(grad)

print(input.neurons[0].weights)
print(input.neurons[1].weights)
print(input.neurons[2].weights)

print(hidden.neurons[0].weights)
print(hidden.neurons[1].weights)

print(output.neurons[0].weights)
print(output.neurons[1].weights)

print(f"total error: {output.total_error([[1,4,5]],[[0.1,0.05]])}")
"""
for i in range(10000):
    output.forward_propagation([1,4,5])
    grad = output.gradient([0.1,0.05])
    output.descent_all(grad)

print(f"total error: {output.total_error([[1,4,5]],[[0.1,0.05]])}")

#print(output.gradient([2,3],[1]))
"""
print("-------------------------train--------------------------")
print(f"total error: {output.total_error([[1,4,5],[6,3,2]],[[0.1,0.05],[0.3,0.05]])}")
output.train([[1,4,5],[6,3,2]],[[0.1,0.05],[0.3,0.05]], 1000, 10)
print(f"total error: {output.total_error([[1,4,5],[6,3,2]],[[0.1,0.05],[0.3,0.05]])}")