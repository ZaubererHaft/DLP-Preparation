import math
import Neuron as nn


i1 = nn.Neuron("i1",[],[1], nn.id)
i2 = nn.Neuron("i2",[],[1], nn.id)

h1 = nn.Neuron("h1",[i1, i2],[0.15,0.25], nn.sigmoid, 0.35)
h2 = nn.Neuron("h2",[i1, i2],[0.2,0.3], nn.sigmoid, 0.35)

print(h1.forward_propagation([0.05,0.10]))

#input = nn.Neuron("X", [], [1], nn.id)
#hidden = nn.Neuron("H", [input], [1])
#out = nn.Neuron("O", [hidden], [1])

#print(input.predict([0.75]))
#print(hidden.predict([0.75]))

#print(out.forward_propagation([0.75]))