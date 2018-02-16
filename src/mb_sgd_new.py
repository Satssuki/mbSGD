from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.kernel_approximation import RBFSampler
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from csv import reader
import argparse


# Find the min and max values for each column
def find_minmax(data):
    minmax = list()
    for i in range(len(data[0])):
        col_values = [row[i] for row in data]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def minmax_tranform(data, minmax):
    for row in data:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return data

# normailize using min-max scale
def minmax_fit_tranform(data):
    data_minmax = find_minmax(data)
    data = minmax_tranform(data, data_minmax)
    return data, data_minmax

# calculcate new features using polynomial transformation
def polynomial_transformation(X):

    # poly = PolynomialFeatures(2)
    # return poly.fit_transform(X)

    result = list()
    index_result = list()

    for record in X:
        record_result = list()

        record_result.append(1)

        for i in range(len(X[0])):
            record_result.append(record[i])

        ind_data = list()

        index_feat = list()

        for i in range(len(X[0])):
            for j in range(len(X[0])):
                if [i, j] not in ind_data and [j, i] not in ind_data:
                    ind_data.append([i, j])
                    index_feat.append([record[i], record[j]])

        for w in index_feat:
            record_result.append(float(w[0] * w[1]))
        
        result.append(record_result)

    return np.array(result)

# calculcate new features using rbf kernel transformation
def rbf_transformation(X, gamma=1):
    rbf_feature = RBFSampler(gamma=gamma)
    trainX = rbf_feature.fit_transform(X)
    return trainX

# merge features from labes
def merge_labels_features(data_x, data_y):
    data = np.c_[data_x, data_y]
    return data

# downloading data from a specific url and check if label column is the last one
def download_data(url, label_index_is_last=True):
    pandas_data = pd.read_csv(url)
    x, y = seperate_labels_features(
        pandas_data.values, label_index_is_last=label_index_is_last)
    data = merge_labels_features(x, y)
    return np.array(data)

# seperate features from labes
def seperate_labels_features(data, label_index_is_last=True):
    data_x = []
    data_y = []

    if(label_index_is_last):
        data_x = data[:, 0:(len(data[0]) - 1)]
        data_y = data[:, -1]
    else:
        data_x = data[:, 1:(len(data[0]))]
        data_y = data[:, 0]

    return data_x, data_y

# https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes
def diabetes_data():
    return download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data', True)

# https://archive.ics.uci.edu/ml/datasets/wine
def wine_data():
    return download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', False)

# https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data
def balance_scale_data():
    return download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data', label_index_is_last=False)

# https://archive.ics.uci.edu/ml/datasets/Wholesale+customers
def wholesales_customers_data():
    return download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv', label_index_is_last=False)

# fetch a new batch for features and labels using a specific batch size
def next_batch(data_x, data_y, batch_size):
    for i in np.arange(0, data_x.shape[0], batch_size):
        yield (data_x[i:i + batch_size], data_y[i:i + batch_size])

# sigmoid function
def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))

# this function uses a set of features and predict a label using coefficients
def predict(row, coefficients):
    yhat = 0
    for i in range(len(row)):
        yhat += coefficients[i] * row[i]

    return sigmoid_activation(yhat)

# just a calculation get accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0

    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) * 100.0

# shuffling the features and labels
def shuffle_data(data_x, data_y):
    data = merge_labels_features(data_x, data_y)
    np.random.shuffle(data)
    return seperate_labels_features(data)

# actual the core algorthm to train our data
def train(trainX, testX, y_train, y_test, number_of_epochs, alpha, batchSize, l2=1.0):
 
    # set a "random value" to init coefficients
    W = np.random.uniform(size=(trainX.shape[1],))

    # loop through epochs
    for epoch in np.arange(0, number_of_epochs):

        # shuffling data before each epoch
        shuffled_data_x, shuffled_data_y = shuffle_data(trainX, y_train)
        #shuffled_data_x, shuffled_data_y = trainX, y_train
        for (batchX, batchY) in next_batch(shuffled_data_x, shuffled_data_y, batchSize):
            # get predicted values from sigmoid function
            preds = sigmoid_activation(batchX.dot(W))
            # regularization using l2
            # W = (W * (1 - (alpha * l2))) - alpha * np.dot((preds - batchY).T, batchX)
            # calculate error
            error = preds - batchY
            # batch loass is the square value of error
            loss = np.sum(error ** 2)
            gradient = batchX.T.dot(error) / batchX.shape[0]

            # update coefficients
            W += -alpha * gradient

    predictions = list()

    for i in range(len(testX)):
        predicted = predict(testX[i], W)
        predicted = round(predicted)
        predictions.append(predicted)

    accuracy = accuracy_metric(y_test, predictions)

    return accuracy, W

# get the number of labels our dataset has
def get_classes(labels):
    classes = list(labels)
    unique_classes = np.unique(classes)
    print 'unique_classes: ' + str(unique_classes)

    if len(unique_classes) == 2:
        return [classes]

    new_labels = list()
    for i in range(len(unique_classes) - 1):
        class_tranformation = list()
        for j in range(len(classes)):

            if(classes[j] == unique_classes[i]):
                class_tranformation.append(1)
            else:
                class_tranformation.append(0)

    new_labels.append(class_tranformation)

    return new_labels

# choosing dataset from user input
def load_data(input):
    dataset = list()
    if input.lower() == 'wine':
        print 'loading wine data...'
        dataset = wine_data()
    elif input.lower() == 'wholesales':
        print 'loading wholesales customers data...'
        dataset = wholesales_customers_data()
    elif input.lower() == 'diabetes':
        print 'loading diabetes data...'
        dataset = diabetes_data()
    elif input.lower() == 'balance':
        print 'loading balance scale data...'
        dataset = balance_scale_data()
    return dataset

# read user arguments
def get_parameters():

    ap = argparse.ArgumentParser()
    
    ap.add_argument("-e", "--epochs", type=int, default=2000, help="# of epochs")
    ap.add_argument("-a", "--alpha", type=float, default=0.001, help="learning rate")
    ap.add_argument("-b", "--batch_size", type=int, default=50, help="size of SGD mini-batches")
    ap.add_argument("-t", "--tranformation_type", type=str, default="pol", help="tranformation type 'pol' or 'rbf'")
    ap.add_argument("-f", "--folds", type=int, default=5, help="# of folds")
    ap.add_argument("-l", "--l2", type=float, default=1.0, help="l2")
    ap.add_argument("-g", "--gamma", type=float, default=1.0, help="gamma parameter for RBF tranformation")
    ap.add_argument("-d", "--dataset", type=str, default='diabetes', help="which dataset to use for training, options: 'wine', 'wholesales', 'diabetes', 'balance'")
    args = vars(ap.parse_args())

    n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, data =args["epochs"], args["alpha"], args["batch_size"], args["tranformation_type"], args["folds"], args["l2"], args["gamma"], args["dataset"]

    print 'epochs: ' + str(n_epochs)
    print 'learning rate: ' + str(l_rate)
    print 'batch size: ' + str(batch_size)
    print 'tranformation type: ' + tranformation_type
    print 'number of folds: ' + str(number_of_folds)
    print 'l2: ' + str(l_2)
    print 'gamma: ' + str(gamma)
    print 'dataset: ' + str(data)

    return n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, data

# cross validation method to find model
def cross_validation(n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, X, y):
    
    print 'data length: ' + str(len(X))

    X_train, X_test, y_train, y_test = [], [], [], []

    kf = KFold(n_splits=number_of_folds)

    accuracies = list()
    coeffecients = list()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        trainX, minmax = minmax_fit_tranform(X_train)
        testX = minmax_tranform(X_test, minmax)

        print 'Tranformation ' + tranformation_type + '...'

        if(tranformation_type == 'rbf'):
            trainX = rbf_transformation(trainX, gamma)
            testX = rbf_transformation(testX, gamma)
        else:
            trainX = polynomial_transformation(trainX)
            testX = polynomial_transformation(testX)
            
        print 'trainX: ' + str(len(trainX[0]))
        print 'testX: ' + str(len(testX[0]))

        accuracy, coeffecient = train(trainX, testX, y_train, y_test, n_epochs, l_rate, batch_size, l2=l_2)
        accuracies.append(accuracy)
        coeffecients.append(coeffecient)

    print coeffecients
    print 'accuracies: ' + str(accuracies)
    print 'avg accuracy: ' + str(np.mean(accuracies))

n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, data = get_parameters()
n_epochs = 2000
l_rate = 0.01 
batch_size = 50
tranformation_type = 'pol'
number_of_folds = 5
l_2 = 1.0
gamma = 1.0
data = 'diabetes'
# load data
dataset = load_data(data)
X, y = seperate_labels_features(dataset)
# X, y = make_blobs(n_samples=700, n_features=2, centers=2, cluster_std=2.5, random_state=95)

cross_validation(n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, X, y)