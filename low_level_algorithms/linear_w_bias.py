import pandas as pd
from matplotlib import pyplot as plt

def cost_function(radio, sales, weight, bias):
    companies = len(radio)
    total_error = 0.0
    for i in range(companies):
        total_error += (sales[i] - (weight*radio[i] + bias))**2
    return total_error / companies

def update_weights(radio, sales, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    companies = len(radio)

    for i in range(companies):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv += -2*radio[i] * (sales[i] - (weight*radio[i] + bias))

        # -2(y - (mx + b))
        bias_deriv += -2*(sales[i] - (weight*radio[i] + bias))

    # We subtract because the derivatives point in direction of steepest ascent
    weight -= (weight_deriv / companies) * learning_rate
    bias -= (bias_deriv / companies) * learning_rate

    return weight, bias


def train(radio, sales, weight, bias, learning_rate, iters):
    cost_history = []


    for i in range(iters):
        weight,bias = update_weights(radio, sales, weight, bias, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(radio, sales, weight, bias)
        cost_history.append([i, cost])

        # Log Progress
        #if i % 5 == 0:
        print ("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost))

    return pd.DataFrame(data=cost_history, columns=['iteration','cost'])

ds = train ([1,2,3,4,5], [3,4,2,4,5], 0, 0, 0.02, 51)

#plot dataset
plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Cost")

plt.plot(ds['cost'], label="Loss")
plt.legend()
plt.show()
