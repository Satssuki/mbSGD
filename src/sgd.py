import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import dataloaders
from random import seed
from random import randrange
from csv import reader
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import FoldWrapper


def sigmoid_activation(x):
    # compute and return the sigmoid activation value for a
    # given input value
    return 1.0 / (1 + np.exp(-x))


def next_batch(data_x, data_y, batch_size):
    # loop over our dataset `X` in mini-batches of size `batchSize`
    for i in np.arange(0, data_x.shape[0], batch_size):
        # yield a tuple of the current batched data and labels
        yield (data_x[i:i + batch_size], data_y[i:i + batch_size])


def mb_sgd(data_x, data_y, epochs, batch_size):
    W = np.random.uniform(size=(X.shape[1],))

    # initialize a list to store the loss value for each epoch
    lossHistory = []

    # loop over the desired number of epochs
    for epoch in np.arange(0, epochs):
        # initialize the total loss for the epoch
        epochLoss = []

        # loop over our data in batches
        for (batchX, batchY) in next_batch(data_x, data_y, batch_size):
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
            W += -alpha * gradient

        # update our loss history list by taking the average loss
        # across all batches
        lossHistory.append(np.average(epochLoss))

    # compute the line of best fit by setting the sigmoid function
    # to 0 and solving for X2 in terms of X1
    Y = (-W[0] - (W[1] * data_x)) / W[2]
    return Y, W, lossHistory


# Split a dataset into k folds
def cross_validation_split_in_folds(data_x, data_y, n_folds):
    dataset_in_folds = list()
    data_x_copy = list(data_x)
    data_y_copy = list(data_y)
    fold_size = int(len(data_y) / n_folds)

    for i in range(n_folds):
        fold_x = list()
        fold_y = list()
        while len(fold_x) < fold_size:
            index = randrange(len(data_x_copy))
            fold_x.append(data_x_copy.pop(index))
            fold_y.append(data_y_copy.pop(index))

        dataset_in_folds.append(FoldWrapper(fold_x, fold_y))

    return dataset_in_folds


# Split a dataset into k folds
def cross_validation_split(data, n_folds):
    dataset_split = list()
    dataset_copy = list(data)
    fold_size = int(len(data) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(data_x, data_y, algorithm, n_folds, *args):
    folds = cross_validation_split_in_folds(data_x, data_y, n_folds)
    scores = list()
    coef = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted, coef = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores, coef


number_of_epochs = 100
alpha = 0.01
batchSize = 32
number_of_folds = 5

# generate a 2-class classification problem with 400 data points,
# where each data point is a 2D feature vector
# (X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)
(X, y) = dataloaders.decision_data()
print X
print y
# insert a column of 1's as the first entry in the feature
# vector -- this is a little trick that allows us to treat
# the bias as a trainable parameter *within* the weight matrix
# rather than an entirely separate variable
X = np.c_[np.ones((X.shape[0])), X]

# initialize our weight matrix such it has the same number of
# columns as our input features
print("[INFO] starting training...")

# Y, W, lossHistory = evaluate_algorithm(X, y, mb_sgd, number_of_folds, number_of_epochs, batchSize)
Y, W, lossHistory = mb_sgd(X, y, number_of_epochs, batchSize)

print W
# plot the original data along with our line of best fit
plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
plt.plot(X, Y, "r-")

# construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(np.arange(0, number_of_epochs), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()