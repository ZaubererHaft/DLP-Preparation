import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#sums for a specific example
def sum(weights, features, bias):
    sum = 0
    for i in range(len(weights)):
        sum += weights[i] * features[i]
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

def gradient(weights, features, bias, labels):
    n = len(weights)
    gradients = [0] * n

    for i in range(n):
        weight_deriv = 0

        for j in range(len(features)):
            numerator1 = features[j][i] * labels[j]
            denominator1 = math.exp(sum(weights, features[j], bias)) + 1
            term1 = - numerator1 / denominator1

            numerator2 = features[j][i] * (labels[j] - 1) * math.exp(sum(weights, features[j], bias))
            denominator2 = math.exp(sum(weights, features[j], bias)) + 1
            term2 = - numerator2 / denominator2

            weight_deriv += term1 + term2
        
        gradients[i] = weight_deriv

    return gradients

def bias_gradient(weights, features, bias, labels):
    bias_deriv = 0
    for i in range(len(features)):
        numerator1 = labels[i]
        denominator1 = math.exp(sum(weights,features[i], bias)) + 1
        term1 = - numerator1 / denominator1

        numerator2 = (labels[i] - 1) * math.exp(sum(weights,features[i], bias))
        denominator2 = math.exp(sum(weights,features[i], bias)) + 1
        term2 = - numerator2 / denominator2

        bias_deriv += term1 + term2

    return bias_deriv

def descent(gradients, grad_bias, weights, bias, learning_rate):
    for i in range(len(weights)):
        weights[i] -= gradients[i]* learning_rate

    bias -= grad_bias  * learning_rate
    return weights, bias

def train(weights, features, bias, labels, learning_rate, iters):

    for i in range(iters):
        gradients = gradient(weights, features, bias, labels)
        grad_bias = bias_gradient(weights, features, bias, labels)

        weights, bias = descent(gradients, grad_bias, weights, bias, learning_rate)

        cost = log_cost(weights, features, bias, labels)

        # Log Progress
        if i % 10 == 0:
            print (f"iter={i}    weights={weights}    bias={bias}    cost={cost}")


#print(hypothesis([2,-1,5],[0,10,2], 1))
#print(log_cost([0],[[1],[2],[3]],0,[0.3,0.1,0.6]))

#print(log_cost([0,0],[[1,4],[2,5],[3,6]],0,[0.3,0.1,0.6]))
#print(gradient([0,0],[[1,4],[2,5],[3,6]],0,[0.3,0.1,0.6]))
train([0,0],[[1,4],[2,5],[3,6]],0,[0.3,0.1,0.6],0.1,5001)

#train([0],[[1],[2],[3],[4],[5],[6],[7],[8]],0,[0.3,0.1,0.6,0.2,0.7,0.2,0.5,0.6], 0.001, 1001)
