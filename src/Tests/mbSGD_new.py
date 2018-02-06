# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
from sklearn.datasets.samples_generator import make_blobs
from math import exp
import numpy as np


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(data, column):
    for row in data:
        row[column] = float(row[column].strip())


# Find the min and max values for each column
def dataset_minmax(data):
    minmax = list()
    for i in range(len(data[0])):
        col_values = [row[i] for row in data]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(data, minmax):
    for row in data:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


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


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(data, algorithm, n_folds, *args):
    folds = cross_validation_split(data, n_folds)
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


# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + np.exp(-yhat))


# Make a prediction with coefficients
def batch_predict(rows, coefficients):
    yhat = coefficients[0]
    x = rows.dot(yhat)
    return 1.0 / (1.0 + np.exp(-x))


def next_batch(data, batchSize):
    # loop over our dataset in mini-batches of size `batchSize`
    for i in np.arange(0, len(data), batchSize):
        yield data[i:i + batchSize]


# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch, batch_size=32):
    data_set = np.array(train)
    coef = [0.0 for i in range(len(train[0]))]

    # for row in train:
    #     yhat = predict(row, coef)
    #     error = row[-1] - yhat
    #     coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
    #     for i in range(len(row)-1):
    #         coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]

    for epoch in range(n_epoch):
        np.random.shuffle(data_set)
        # loop over our data in batches
        for batch_data in next_batch(data_set, batch_size):
            batch_x = batch_data[0:len(batch_data[0]) - 2]
            batch_y = batch_data[:, -1]

            # TODO
            yhat = batch_predict(batch_data, coef)
            batch_error = batch_data[:, -1] - yhat
            yhat_avg = batch_data[:, -1].dot(batch_error) / batch_data.shape[0]
            
            coef[0] = coef[0] + l_rate * batch_error * yhat_avg * (1.0 - yhat_avg)
            for i in range(len(batch_data[0])-1):
                coef[i + 1] = coef[i + 1] + l_rate * batch_error * yhat_avg * (1.0 - yhat_avg) * batch_data[:i]

    return coef


# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch, batch_size):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch, batch_size)
    print 'coef'
    print coef
    for row in test:
        yhat = predict(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return predictions, coef


def get_random_data():
    data = list()
    (X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)

    for i in range(len(X)):
        record = list()
        record.append(X[i][0])
        record.append(X[i][1])
        record.append(y[i])
        data.append(record)
    return data


# https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes
def get_csv_data():
    file_name = 'C:\Data\pima-indians-diabetes.csv'
    data = load_csv(file_name)

    for ii in range(len(data[0])):
        str_column_to_float(data, ii)

    return data


# https://stats.idre.ucla.edu/r/dae/logit-regression/
def get_csv_decision_data():
    file_name = 'C:\Users\cfilip09\Downloads\\binary.csv'
    data = load_csv(file_name)

    for ii in range(len(data[0])):
        str_column_to_float(data, ii)

    for record in data:
        record.reverse()

    return data


# https://archive.ics.uci.edu/ml/datasets/Haberman's%2BSurvival
def get_csv_haberman_data():
    file_name = 'C:\Data\haberman\haberman.csv'
    data = load_csv(file_name)

    for ii in range(len(data[0])):
        str_column_to_float(data, ii)

    return data


def normalize_data(data):
    # scaler = StandardScaler()
    # scaler.fit(data)
    # data_scaled = scaler.transform(data)
    # return data_scaled.tolist()
    data_minmax = dataset_minmax(data)
    normalize_dataset(data, data_minmax)
    return data


# Test the logistic regression algorithm on the diabetes dataset
seed(1)
# load and prepare data

dataset = get_csv_data()
# dataset = get_csv_haberman_data()
# dataset = get_csv_decision_data()
dataset = normalize_data(dataset)
print dataset[1]
# evaluate algorithm
number_of_folds = 5
learning_rate = 0.1
number_of_epochs = 100
batchSize = 100

scores, coeff = evaluate_algorithm(dataset, logistic_regression, number_of_folds, learning_rate, number_of_epochs, batchSize)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
print coeff
