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

# actually transform data
def transform_data(X, tranformation_type, gamma):
    trainX = X
    if(tranformation_type == 'rbf'):
        trainX = rbf_transformation(X, gamma)
    else:
        trainX = polynomial_transformation(X)

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
def mb_sgd(trainX, y_train, number_of_epochs, alpha, batchSize, l2=1.0):
 
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

    return W

# use test set to get how accurate is our model
def test(testX, y_test, W):
    predictions = list()

    for i in range(len(testX)):
        predicted = predict(testX[i], W)
        predicted = round(predicted)
        predictions.append(predicted)

    accuracy = accuracy_metric(y_test, predictions)
    return accuracy

# get the number of labels our dataset has
def get_classes(labels):
    classes = list(labels)
    unique_classes = np.unique(classes)
    print 'unique_classes: ' + str(unique_classes)

    new_labels = list()
    
    for i in range(len(unique_classes)):
        class_tranformation = list()
        for j in range(len(classes)):
            if(classes[j] == unique_classes[i]):
                class_tranformation.append(1)
            else:
                class_tranformation.append(0)
        new_labels.append((unique_classes[i], np.array(class_tranformation)))
    
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
    ap.add_argument("-m", "--mode", type=str, default='cross', help="choose if either to evaluate metaparameters using crossvalidation or get accuracy, options: 'cross', 'train'")
    args = vars(ap.parse_args())

    n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, data, mode =args["epochs"], args["alpha"], args["batch_size"], args["tranformation_type"], args["folds"], args["l2"], args["gamma"], args["dataset"], args["mode"]

    print 'epochs: ' + str(n_epochs)
    print 'learning rate: ' + str(l_rate)
    print 'batch size: ' + str(batch_size)
    print 'tranformation type: ' + tranformation_type
    print 'number of folds: ' + str(number_of_folds)
    print 'l2: ' + str(l_2)
    print 'gamma: ' + str(gamma)
    print 'dataset: ' + str(data)
    print 'mode: ' + mode

    return n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, data, mode

# cross validation method to find model
def cross_validation(n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, X, y):
    
    print 'cross validation'

    X_train, X_test, y_train, y_test = [], [], [], []

    kf = KFold(n_splits=number_of_folds)

    accuracies = list()
    coeffecients = list()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        trainX, minmax = minmax_fit_tranform(X_train)
        testX = minmax_tranform(X_test, minmax)

        trainX = transform_data(trainX, tranformation_type, gamma)
        testX = transform_data(testX, tranformation_type, gamma)

        coeffecient = mb_sgd(trainX, y_train, n_epochs, l_rate, batch_size, l2=l_2)
        accuracy = test(testX, y_test, coeffecient)

        accuracies.append(accuracy)
        coeffecients.append(coeffecient)

    print 'accuracies: ' + str(accuracies)
    print 'avg accuracy: ' + str(np.mean(accuracies))

# cross validation method to find model
def multiclass_cross_validation(n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, X, ovr_y, y):
    
    print 'multiclass cross validation'

    class_coeffecients = list()
    for class_y in ovr_y:
        X_train, X_test, y_train, y_test = [], [], [], []

        kf = KFold(n_splits=number_of_folds)

        accuracies = list()
        coeffecients = list()

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = class_y[1][train_index], class_y[1][test_index]

            trainX, minmax = minmax_fit_tranform(X_train)
            testX = minmax_tranform(X_test, minmax)

            trainX = transform_data(trainX, tranformation_type, gamma)
            testX = transform_data(testX, tranformation_type, gamma)

            coeffecient = mb_sgd(trainX, y_train, n_epochs, l_rate, batch_size, l2=l_2)
            accuracy = test(testX, y_test, coeffecient)

            accuracies.append(accuracy)
            coeffecients.append(coeffecient)
        
        class_coeffecients.append((class_y[0], coeffecients[0]))

        print 'class ' + str(class_y[0]) + ' accuracies: ' + str(accuracies)
        print 'class ' + str(class_y[0]) + ' avg accuracy: ' + str(np.mean(accuracies))

    x_in, minmax = minmax_fit_tranform(X)
    x_in = transform_data(x_in, tranformation_type, gamma)

    predictions = list()

    for ii in range(len(y)):
        label_result_index = 0
        label_result = 0
        
        for class_label in class_coeffecients:
            predicted = predict(x_in[ii], class_label[1])

            if label_result < predicted:
                label_result = predicted
                label_result_index = class_label[0]
            
        predictions.append(label_result_index)
            
    accuracy = accuracy_metric(y, predictions)

    print accuracy

# train  
def train(n_epochs, l_rate, batch_size, tranformation_type, l_2, gamma, X, y):
    trainX, minmax = minmax_fit_tranform(X)
    trainX = transform_data(trainX, tranformation_type, gamma)
    return mb_sgd(trainX, y, n_epochs, l_rate, batch_size, l2=l_2)

# multiclass train  
def multiclass_train(n_epochs, l_rate, batch_size, tranformation_type, l_2, gamma, X, ovr_y):
    class_coeffecients = list()
    for class_y in ovr_y:
        coef = train(n_epochs, l_rate, batch_size, tranformation_type, l_2, gamma, X, class_y[1])
        class_coeffecients.append((class_y[0], coef))
    return class_coeffecients

n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, data, mode = get_parameters()
n_epochs = 2000
l_rate = 0.01 
batch_size = 50
tranformation_type = 'pol'
number_of_folds = 5
l_2 = 1.0
gamma = 1.0
data = 'diabetes'
mode = 'cross'
leave_out = 0.05

# load data
dataset = load_data(data)
X, y = seperate_labels_features(dataset)
# X, y = make_blobs(n_samples=700, n_features=2, centers=2, cluster_std=2.5, random_state=95)

index_to_leave_out = int(round(len(X) * leave_out))

print 'data length: ' + str(len(X))
print 'leaving out : ' + str(index_to_leave_out) + \
    ' records (' + str(leave_out * 100) + '%)'

# leaving some data out of model
X_out = X[-index_to_leave_out:]
y_out = y[-index_to_leave_out:]

X_in = X[:-index_to_leave_out]
y_in = y[:-index_to_leave_out]


classes = get_classes(y_in)

if mode == 'cross':
    if len(classes) == 2:
        cross_validation(n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, X_in, y_in)
    else:
        multiclass_cross_validation(n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, X_in, classes, y_in)
else:

    print 'evaluate'
    out_X, minmax = minmax_fit_tranform(X_out)

    if len(classes) == 2:
        coeffecients = train(n_epochs, l_rate, batch_size, tranformation_type, l_2, gamma, out_X, y_out)
        predictions = list()

        for i in range(len(out_X)):
            predicted = predict(out_X[i], coeffecients)
            predicted = round(predicted)
            predictions.append(predicted)

        accuracy = accuracy_metric(y_out, predictions)
        print accuracy
    else:
        coeffecients =multiclass_train(n_epochs, l_rate, batch_size, tranformation_type,  l_2, gamma, out_X, classes)