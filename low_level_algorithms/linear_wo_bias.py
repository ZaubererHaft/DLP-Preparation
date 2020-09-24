def cost_function(radio, sales, weight):
    companies = len(radio)
    total_error = 0.0
    for i in range(companies):
        total_error += (sales[i] - (weight*radio[i]))**2
    return total_error / companies

def gradient(radio, sales, weight):
    weight_deriv = 0
    bias_deriv = 0
    companies = len(radio)

    for i in range(companies):
        # Calculate partial derivatives
        # 2x_i * (x_im-y_i)
        weight_deriv += 2*radio[i] * (weight*radio[i] - sales[i])

    return weight_deriv / companies

def train(radio, sales, weight, learning_rate, iters):
    cost_history = []

    for i in range(iters):

        grad = gradient(radio, sales, weight)
        #gradient descent
        weight -= (grad * learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(radio, sales, weight)
        cost_history.append(cost)

        # Log Progress
        # if i % 10 == 0:
        print ("iter={:d}    weight={:.2f}     cost={:.5}".format(i, weight, cost))

    return weight , cost_history

train ([1,2,3,4,5], [3,4,2,4,5], 30, 0.02, 20)