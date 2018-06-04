from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from csv import reader
import argparse


def merge_labels_features(data_x, data_y):
    """ Merge features from labels """
    data = np.c_[data_x, data_y]
    return data

def seperate_labels_features(data, label_index_is_last=True):
    """ Seperate features from labes """
    data_x = []
    data_y = [] 

    if(label_index_is_last):
        data_x = data[:, 0:(len(data[0]) - 1)]
        data_y = data[:, -1]
    else:
        data_x = data[:, 1:(len(data[0]))]
        data_y = data[:, 0]

    return data_x, data_y

def find_minmax(data):
    """ Find the min and max values for each column """
    minmax = list()
    for i in range(len(data[0])):
        col_values = [row[i] for row in data]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def minmax_tranform(data, minmax):
    """ Rescale dataset columns to the range 0-1 """
    for row in data:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return data

def minmax_fit_tranform(data):
    """ Normailize using min-max scale """
    data_minmax = find_minmax(data)
    data = minmax_tranform(data, data_minmax)
    return data, data_minmax

def polynomial_transformation(X):
    """ Calculcate new features using polynomial transformation """

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

def rbf_transformation(X, gamma=1):
    """ Calculcate new features using rbf kernel transformation """
    # return rbf_kernel(X, None, gamma)

    K = euclidean_distances(X, X, squared=True)
    K *= -gamma
    np.exp(K, K)
    return K

def transform_data(X, tranformation_type, gamma):
    """ Actually transform data using user selection """
    trainX = X
    if(tranformation_type == 'rbf'):
        trainX = rbf_transformation(X, gamma)
    else:
        trainX = polynomial_transformation(X)

    return trainX

def download_data(url, label_index_is_last=True):
    """ Downloading data from a specific url and check if label column is the last one """
    pandas_data = pd.read_csv(url)
    x, y = seperate_labels_features(
        pandas_data.values, label_index_is_last=label_index_is_last)
    data = merge_labels_features(x, y)
    return np.array(data)

def diabetes_data():
    """ Data from https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes """
    return download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data', True)

def wine_data():
    """ Data from https://archive.ics.uci.edu/ml/datasets/wine """
    return download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', False)

def balance_scale_data():
    """ https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data """
    data = download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data', label_index_is_last=False)
    X, y =seperate_labels_features(data)
    new_y = list()
    for yy in y:
        if(yy == 'B'):
            new_y.append(0.0)
        elif (yy == 'L'):
            new_y.append(1.0)
        elif (yy == 'R'):
            new_y.append(2.0)

    return merge_labels_features(X, np.array(new_y))

def wholesales_customers_data():
    """ https://archive.ics.uci.edu/ml/datasets/Wholesale+customers """
    return download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv', label_index_is_last=False)

def next_batch(data_x, data_y, batch_size):
    """ Fetch a new batch for features and labels using a specific batch size """
    for i in np.arange(0, data_x.shape[0], batch_size):
        yield (data_x[i:i + batch_size], data_y[i:i + batch_size])

def sigmoid_activation(x):
    """ Sigmoid function """
    return 1.0 / (1 + np.exp(-x))

def predict(row, coefficients):
    """ this function uses a set of features and predict a label using coefficients """
    yhat = 0
    for i in range(len(row)):
        yhat += coefficients[i] * row[i]

    return sigmoid_activation(yhat)

def accuracy_metric(actual, predicted):
    """ Just a calculation get accuracy percentage """
    correct = 0

    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) * 100.0

def shuffle_data(data_x, data_y):
    """ shuffling the features and labels """
    data = merge_labels_features(data_x, data_y)
    np.random.shuffle(data)
    return seperate_labels_features(data)

def mb_sgd(trainX, y_train, number_of_epochs, alpha, batchSize, l2=1.0):
    """ Actually the core algorthm to train our data """
 
    # set a "random value" to init coefficients
    W = np.random.uniform(size=(trainX.shape[1],))

    # loop through epochs
    for epoch in np.arange(0, number_of_epochs):

        # shuffling data before each epoch
        shuffled_data_x, shuffled_data_y = shuffle_data(trainX, y_train)

        for (batchX, batchY) in next_batch(shuffled_data_x, shuffled_data_y, batchSize):
            # get predicted values from sigmoid function
            preds = sigmoid_activation(batchX.dot(W))
            # calculate error
            error = preds - batchY
            # batch loass is the square value of error
            loss = np.sum(error ** 2)

            # update coefficients
            # gradient = batchX.T.dot(error) / batchX.shape[0]
            #W += -alpha * gradient
            
            # regularization
            W = (W * (1 - (alpha * l2))) - alpha * np.dot(error.T, batchX)

    return W

def test(testX, y_test, W):
    """ Use test set to get how accurate is our model """
    predictions = list()

    for i in range(len(testX)):
        predicted = predict(testX[i], W)
        predicted = round(predicted)
        predictions.append(predicted)

    accuracy = accuracy_metric(y_test, predictions)
    return accuracy

def get_classes(labels):
    """ Get the number of labels in our dataset. It uses the OVR approach """
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

def load_data(input):
    """ Choosing dataset using user input """
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

    X, y = seperate_labels_features(dataset)
    return X, y

def cross_validation(n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, X, y):
    """ Cross validation method to find model accuracy. Two classes dataset."""

    print 'cross validation'

    X_train, X_test, y_train, y_test = [], [], [], []

    minmax = find_minmax(X)
    X = minmax_tranform(X, minmax)
    X = transform_data(X, tranformation_type, gamma)
    
    accuracies = list()
    coeffecients = list()

    kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=4)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        trainX = X_train
        testX = X_test

        # trainX, minmax = minmax_fit_tranform(X_train)
        # testX = minmax_tranform(X_test, minmax)

        # trainX = transform_data(trainX, tranformation_type, gamma)
        # testX = transform_data(testX, tranformation_type, gamma)

        coeffecient = mb_sgd(trainX, y_train, n_epochs, l_rate, batch_size, l2=l_2)
        accuracy = test(testX, y_test, coeffecient)

        accuracies.append(accuracy)
        coeffecients.append(coeffecient)

    print 'accuracies: ' + str(accuracies)
    print 'avg accuracy: ' + str(np.mean(accuracies))

def multiclass_cross_validation(n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, X, ovr_y, y):
    """ Cross validation method to find model accuracy. Multiclass dataset."""
    print 'multiclass cross validation'

    class_coeffecients = list()

    minmax = find_minmax(X)

    X = minmax_tranform(X, minmax)
    X = transform_data(X, tranformation_type, gamma)

    # print 'minmax' + str(minmax)

    for class_y in ovr_y:
        X_train, X_test, y_train, y_test = [], [], [], []

        kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=4)

        accuracies = list()
        coeffecients = list()

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = class_y[1][train_index], class_y[1][test_index]

            trainX = X_train
            testX = X_test

            # trainX = minmax_tranform(X_train, minmax)
            # testX = minmax_tranform(X_test, minmax)

            # trainX = transform_data(trainX, tranformation_type, gamma)
            # testX = transform_data(testX, tranformation_type, gamma)

            coeffecient = mb_sgd(trainX, y_train, n_epochs, l_rate, batch_size, l2=l_2)
            accuracy = test(testX, y_test, coeffecient)

            accuracies.append(accuracy)
            coeffecients.append(coeffecient)
        
        class_coeffecients.append((class_y[0], coeffecients[0]))

        print 'class ' + str(class_y[0]) + ' accuracies: ' + str(accuracies)
        print 'class ' + str(class_y[0]) + ' avg accuracy: ' + str(np.mean(accuracies))

    # x_in = minmax_tranform(X, minmax)
    # x_in = transform_data(x_in, tranformation_type, gamma)
    x_in = X

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
  
def train(n_epochs, l_rate, batch_size, tranformation_type, l_2, gamma, X, y):
    """ Scale data and train """ 
    trainX, minmax = minmax_fit_tranform(X)
    trainX = transform_data(trainX, tranformation_type, gamma)
    return mb_sgd(trainX, y, n_epochs, l_rate, batch_size, l2=l_2)

def multiclass_train(n_epochs, l_rate, batch_size, tranformation_type, l_2, gamma, X, ovr_y):
    """ Scale data and train for each classifier. Multiclass datasets.""" 
    class_coeffecients = list()
    for class_y in ovr_y:
        coef = train(n_epochs, l_rate, batch_size, tranformation_type, l_2, gamma, X, class_y[1])
        class_coeffecients.append((class_y[0], coef))
    return class_coeffecients

def leave_out_some_data(X, y, percentage_to_leave_out):
    """ Leaving some data out of training in order to use it later """

    index_to_leave_out = int(round(len(X) * percentage_to_leave_out))

    print 'data length: ' + str(len(X))
    print 'leaving out : ' + str(index_to_leave_out) + ' records (' + str(percentage_to_leave_out * 100) + '%)'

    # leaving some data out of model
    X_out = X[-index_to_leave_out:]
    y_out = y[-index_to_leave_out:]

    X_in = X[:-index_to_leave_out]
    y_in = y[:-index_to_leave_out]
    return X_in, X_out, y_in, y_out

def get_parameters():
    """ read user arguments """

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

    # n_epochs = 2000
    # l_rate = 0.01 
    # batch_size = 50
    # tranformation_type = 'pol'
    # number_of_folds = 5
    # l_2 = 1.0
    # gamma = 1.0
    # data = 'diabetes'

    print 'epochs: ' + str(n_epochs)
    print 'learning rate: ' + str(l_rate)
    print 'batch size: ' + str(batch_size)
    print 'tranformation type: ' + tranformation_type
    print 'number of folds: ' + str(number_of_folds)
    print 'l2: ' + str(l_2)
    print 'gamma: ' + str(gamma)
    print 'dataset: ' + str(data)

    return n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, data

def get_results():
    n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, data = get_parameters()
    leave_out = 0.05

    # load data
    X, y = load_data(data)
    
    # from sklearn.datasets.samples_generator import make_blobs
    # X, y = make_blobs(n_samples=700, n_features=2, centers=2, cluster_std=2.5, random_state=95)

    # leaving some data out of training
    X_in, X_out, y_in, y_out = leave_out_some_data(X, y, leave_out)
    classes = get_classes(y_in)

    if len(classes) == 2:
        cross_validation(n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, X_in, classes[0][1])
    else:
        multiclass_cross_validation(n_epochs, l_rate, batch_size, tranformation_type, number_of_folds, l_2, gamma, X_in, classes, y_in)

if __name__ == "__main__":
    get_results()
