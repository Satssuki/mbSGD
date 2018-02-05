import matplotlib.pyplot as plt
import numpy as np


def sigmoid_activation(x):
    # compute and return the sigmoid activation value for a
    # given input value
    return 1.0 / (1 + np.exp(-x))


def next_batch(X, y, batch_size):
    # loop over our dataset `X` in mini-batches of size `batchSize`
    for i in np.arange(0, X.shape[0], batch_size):
        # yield a tuple of the current batched data and labels
        yield (X[i:i + batch_size], y[i:i + batch_size])


def mbSGD(epochs, learning_rate, batch_size, X, y):
    W = np.random.uniform(size=(X.shape[1],))

    # initialize a list to store the loss value for each epoch
    lossHistory = []

    # loop over the desired number of epochs
    for epoch in np.arange(0, epochs):
        # initialize the total loss for the epoch
        epochLoss = []

        # loop over our data in batches
        for (batchX, batchY) in next_batch(X, y, batch_size):
            # take the dot product between our current batch of
            # features and weight matrix `W`, then pass this value
            # through the sigmoid activation function
            preds = sigmoid_activation(batchX.dot(W))

            # now that we have our predictions, we need to determine
            # our `error`, which is the difference between our predictions
            # and the true values
            error = preds - batchY

            # given our `error`, we can compute the total loss value on
            # the batch as the sum of squared loss
            loss = np.sum(error ** 2)
            epochLoss.append(loss)

            # the gradient update is therefore the dot product between
            # the transpose of our current batch and the error on the
            # # batch
            gradient = batchX.T.dot(error) / batchX.shape[0]

            # use the gradient computed on the current batch to take
            # a "step" in the correct direction
            W += -learning_rate * gradient

        # update our loss history list by taking the average loss
        # across all batches
        lossHistory.append(np.average(epochLoss))

    # compute the line of best fit by setting the sigmoid function
    # to 0 and solving for X2 in terms of X1
    Y = (-W[0] - (W[1] * X)) / W[2]
    print W
    return Y, lossHistory
    # plot_data(X, Y, epochs, lossHistory)


def plot_data(X, Y, y, epochs, loss_history):
    # plot the original data along with our line of best fit
    plt.figure()
    plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
    plt.plot(X, Y, "r-")

    # construct a figure that plots the loss over time
    fig = plt.figure()
    plt.plot(np.arange(0, epochs), loss_history)
    fig.suptitle("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()


