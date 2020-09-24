import matplotlib.pyplot as plt
import csv


def cost_function(samples, features, weights, bias):
    m = len(samples)
    n = len(weights)

    total_error = 0.0

    for i in range(m):
        sum = 0
        for j in range(n):
            sum += weights[j] * features[j][i]
        sum += bias

        total_error += (samples[i] - sum)**2

    return total_error / m

def gradient(samples, features, weights, bias):
    m = len(samples)
    n = len(weights)

    gradients = [0] * n

    #for each weight
    for i in range(n):
        weight_i_deriv = 0
        bias_deriv = 0

        #for each sample
        for j in range(m):

            #inner sum
            sum = 0
            for k in range(n):
                sum += weights[k] * features[k][j]
            sum += bias

            weight_i_deriv += -2*features[i][j] * (samples[j] - sum)
            bias_deriv += -2*(samples[j] - sum)

        gradients[i] = weight_i_deriv / m

    return gradients, (bias_deriv / m)

def descent(gradients, grad_bias, weights, bias, learning_rate):
    for i in range(len(weights)):
        weights[i] -= gradients[i]* learning_rate

    bias -= grad_bias  * learning_rate
    return weights, bias


def train(samples, features, weights, bias, learning_rate, iters):

    for i in range(iters):
        gradients, grad_bias = gradient(samples, features, weights, bias)
        weights, bias = descent(gradients, grad_bias, weights, bias, learning_rate)

        cost = cost_function(samples, features, weights, bias)

        # Log Progress
        if i % 500 == 0:
            print (f"iter={i}    weights={weights}    bias={bias}    cost={cost}")


#print(cost_function([100,200,300],[[30,40,10],[20,40,100]],[0,0],0))
#print(gradient([100,200,300],[[30,40,10],[20,40,100]],[0,0],0))


#with open('/home/ludwig/Downloads/california_housing_train.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in spamreader:
#         print (', '.join(row))

train([100,200,300],[[30,40,10],[20,40,100]],[0,0],0,0.0001,500000)
#train([3,4,2,4,5],[[1,2,3,4,5]],[0],0,0.02,50)

#print(cost_function([3,4,2,4,5],[[1,2,3,4,5]],[0],0))