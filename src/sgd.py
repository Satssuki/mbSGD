import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import dataloaders
from random import seed
from random import randrange
from csv import reader
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import sys


# Make a prediction with coefficients
def predict(x, yhat, coefficients):
    # yhat = coefficients[0]
    for i in range(len(x)):
        yhat += coefficients[i + 1] * x[i]
    return 1.0 / (1.0 + np.exp(-yhat))


# sigmoid function
def sigmoid_activation(x):
    # compute and return the sigmoid activation value for a
    # given input value
    try:
        return 1.0 / (1 + np.exp(-x))
    except:
        sys.exit(x)


# Split a dataset into k batches
def next_batch(data_x, data_y, batch_size):
    # loop over our dataset `X` in mini-batches of size `batchSize`
    for i in np.arange(0, data_x.shape[0], batch_size):
        # yield a tuple of the current batched data and labels
        yield (data_x[i:i + batch_size], data_y[i:i + batch_size])


# Stochastic Gradient Descent algorithm
def mb_sgd(data_x, data_y, alpha, epochs, batch_size):
    print (data_x.shape[1],)
    W = np.zeros(shape=data_x.shape[1])

    # initialize a list to store the loss value for each epoch
    lossHistory = []

    # loop over the desired number of epochs
    for epoch in np.arange(0, epochs):
        # initialize the total loss for the epoch
        epochLoss = []

        #shuffled_data_x, shuffled_data_y = shuffle_data(data_x, data_y)
        shuffled_data_x, shuffled_data_y = data_x, data_y

        # loop over our data in batches
        for (batchX, batchY) in next_batch(shuffled_data_x, shuffled_data_y, batch_size):
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
def cross_validation_split(data_x, data_y, n_folds):
    data = merge_labels_features(data_x, data_y)
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


# seperate features from labes
def seperate_labels_features(data):
    data_x = data[:, 0:(len(data[0]) - 1)]
    data_y = data[:, 1]
    return data_x, data_y

# merge features from labes
def merge_labels_features(data_x, data_y):
    data = np.c_[data_x, data_y]
    return data


# shuffle data
def shuffle_data(data_x, data_y):
    data = merge_labels_features(data_x, data_y)
    np.random.shuffle(data)
    return seperate_labels_features(data)


# Split a dataset into k folds
def cross_validation_split_in_folds(data_x, data_y, n_folds):
    data = np.c_[data_x, data_y]

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


# Linear Regression Algorithm With Stochastic Gradient Descent
def train_and_test(train_set, test_set, number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature):
    predictions = list()
    x, y = seperate_labels_features(np.array(train_set))
    Y, coef, lossHistory = logistic_regression_mbSGD(x, y, number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature)

    # print 'train_set'
    print train_set
    # sys.exit(0)

    for test_x, test_y in seperate_labels_features(np.array(train_set)):
        yhat = predict(test_x, test_y, coef)
        yhat = round(yhat)
        predictions.append(yhat)

    return predictions, coef, lossHistory, Y


def evaluate_algorithm(data_x, data_y, number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature):
    folds = cross_validation_split(data_x, data_y, number_of_folds)
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

        predicted, coef, lossHistory, Y = train_and_test(train_set, test_set, number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores, coef, lossHistory, Y


# actually the Logistic Regression implementation
def logistic_regression_mbSGD(X, y, number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature=False):

    # X, y = seperate_labels_features(np.array(data)) 
    if(add_additional_feature):
        # insert a column of 1's as the first entry in the feature
        # vector -- this is a little trick that allows us to treat
        # the bias as a trainable parameter *within* the weight matrix
        # rather than an entirely separate variable
        X = np.c_[np.power(X[:, 0], 2), X]

    # initialize our weight matrix such it has the same number of
    # columns as our input features
    print("[INFO] starting training...")

    # Y, W, lossHistory = evaluate_algorithm(X, y, mb_sgd, number_of_folds, number_of_epochs, batchSize)
    Y, W, lossHistory = mb_sgd(X, y, alpha, number_of_epochs, batchSize)

    # print W, lossHistory
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

    return Y, W, lossHistory


# generate a 2-class classification problem with 400 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)
#(X, y) = dataloaders.decision_data()
data = merge_labels_features(X, y)

#scores, coef, lossHistory, Y = evaluate_algorithm(X, y, number_of_epochs=100, alpha=0.01,batchSize=32, number_of_folds=5,add_additional_feature=True)
Y, W, lossHistory = logistic_regression_mbSGD(X, y, number_of_epochs=100, alpha=0.01,batchSize=32, number_of_folds=5,add_additional_feature=True)
# print W
