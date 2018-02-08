import matplotlib.pyplot as plt
import dataloaders
import numpy as np
import sys
from random import seed
from random import randrange
from csv import reader
from sklearn.model_selection import KFold
from sklearn.datasets.samples_generator import make_blobs


# Make a prediction with coefficients
def predict(x, yhat, coefficients):
    # yhat = coefficients[0]
    for i in range(len(x)):
        yhat += coefficients[i] * x[i]
    return sigmoid_activation(yhat)


# sigmoid function
def sigmoid_activation(x):
    # compute and return the sigmoid activation value for a
    # given input value
    return 1.0 / (1 + np.exp(-x))
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
    #print (data_x.shape[1],)
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

# merge features from labes
def merge_labels_features(data_x, data_y):
    data = np.c_[data_x, data_y]
    return data


# shuffle data
def shuffle_data(data_x, data_y):
    data = merge_labels_features(data_x, data_y)
    np.random.shuffle(data)
    return dataloaders.seperate_labels_features(data)


# Linear Regression Algorithm With Stochastic Gradient Descent
def train_and_test(train_set, test_set, number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature):
    predictions = list()
    d_x, d_y = dataloaders.seperate_labels_features(train_set)
    Y, coef, lossHistory = mb_sgd(d_x, d_y, alpha, number_of_epochs, batchSize)

    test_set_x, test_set_y = dataloaders.seperate_labels_features(test_set)
    print 'coef: ' + str(coef)
    for i in range(len(test_set_x)):
        test_x, test_y = test_set_x[i], test_set_y[i]
        yhat = predict(test_x, test_y, coef)
        yhat = round(yhat)
        predictions.append(yhat)

    return predictions, coef, lossHistory, Y


def evaluate_algorithm(data, number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature):

    kf = KFold(n_splits=number_of_folds)
    sum = 0

    scores = list()
    coef = list()
    
    for train, test in kf.split(data):

        train_data = np.array(data)[train]
        test_data = np.array(data)[test]

        # print 'train_data'
        # for i in range(50):
        #     print train_data[i]

        # print '--------------------------------------'

        # print 'test_data'
        # for i in range(50):
        #     print test_data[i]

        predicted, coef, lossHistory, Y = train_and_test(train_data, test_data, number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature)
        
        actual_x, actual_y = dataloaders.seperate_labels_features(test_data)
        accuracy = accuracy_metric(actual_y, predicted)

        for i in range(len(actual_y)):
            print str(actual_y[i]) + '/' + str(predicted[i])

        print 'fold score, expected: ' + str(accuracy)
        scores.append(accuracy)

    return scores, coef, lossHistory, Y


def evaluate_algorithm_old(data, number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature):
    folds = cross_validation_split(data, number_of_folds)
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
            #row_copy[-1] = None

        predicted, coef, lossHistory, Y = train_and_test(np.array(train_set), np.array(test_set), number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores, coef, lossHistory, Y


# actually the Logistic Regression implementation
def logistic_regression_mbSGD(data, number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature=False):

    X, y = dataloaders.seperate_labels_features(np.array(data)) 
    if(add_additional_feature):
        # insert a column of 1's as the first entry in the feature
        # vector -- this is a little trick that allows us to treat
        # the bias as a trainable parameter *within* the weight matrix
        # rather than an entirely separate variable
        X = np.c_[np.power(X[:, 0], 2), X]

    # initialize our weight matrix such it has the same number of
    # columns as our input features
    print("[INFO] starting training...")
    
    scores, W, lossHistory, Y = evaluate_algorithm(dataloaders.merge_labels_features(X, y), number_of_epochs, alpha, batchSize, number_of_folds, add_additional_feature)
    print 'scores'
    print scores
    print 'W'
    print W
    print 'Y'
    print Y
    #exit(0)

    # Y, W, lossHistory = mb_sgd(X, y, alpha, number_of_epochs, batchSize)

    # print W, lossHistory
    # plot the original data along with our line of best fit
    plt.figure()
    plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
    # plt.plot(X, Y, "r-")

    # construct a figure that plots the loss over time
    fig = plt.figure()
    plt.plot(np.arange(0, number_of_epochs), lossHistory)
    fig.suptitle("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()

    return scores, W, lossHistory, Y

# data structure |feature1,feature2,..,featuren,  label|
# data = dataloaders.diabetes_data()
#data = dataloaders.wine_data()
data = dataloaders.random_data()

# print 'data' + str(data[0])
scores, coef, lossHistory, Y = logistic_regression_mbSGD(data, number_of_epochs=50, alpha=0.01,batchSize=32, number_of_folds=5,add_additional_feature=True)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
print coef