import math
import Neuron as nn
import Activator as ac


i1 = nn.create_input_neuron("i1")
i2 = nn.create_input_neuron("i2")

h1 = nn.Neuron("h1",[i1, i2],[0.15, 0.2], ac.Sigmoid(), 0.35)
h2 = nn.Neuron("h2",[i1, i2],[0.25, 0.3], ac.Sigmoid(), 0.35)

o1 = nn.Neuron("o1",[h1, h2],[0.40, 0.45], ac.Sigmoid(), 0.60)
o2 = nn.Neuron("o2",[h1, h2],[0.50, 0.55], ac.Sigmoid(), 0.60)

#print(h1.forward_propagation([0.05, 0.10]))
#print(h2.forward_propagation([0.05, 0.10]))

out_o1 = o1.forward_propagation([0.05, 0.10])
out_o2 = o2.forward_propagation([0.05, 0.10])

print(out_o1)
print(out_o2)

print(nn.error(out_o1, 0.01))
print(nn.error(out_o2, 0.99))
