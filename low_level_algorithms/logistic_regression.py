import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#sums for a specific example
def sum(weights, feature, bias):
    sum = 0
    for i in range(len(weights)):
        sum += weights[i] * feature[i]
    return sum + bias

def hypothesis(weights, feature, bias):
    return sigmoid(sum(weights, feature, bias))

def log_cost(weights, features, bias, labels):
    sum = 0
    for i in range(len(labels)):
        label = labels[i]
        hyp = hypothesis(weights, features[i], bias)
        
        sum += -label * math.log(hyp) - (1-label) * math.log(1 - hyp)
    return sum


print(hypothesis([2,-1,5],[0,10,2], 1))

print(log_cost([0],[[1],[2],[3]],0,[0.3,0.1,0.6]))