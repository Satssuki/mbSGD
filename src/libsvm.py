from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm
import pandas
from csv import reader
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# merge features from labes


def merge_labels_features(data_x, data_y):
    data = np.c_[data_x, data_y]
    return data


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


def wine_data():
    return download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', False)


def accuracy_metric(actual, predicted):
    correct = 0

    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) * 100.0


leave_out = 0.2

dataset = wine_data()
np.random.shuffle(dataset)
X, y = seperate_labels_features(dataset)
index_to_leave_out = int(round(len(X) * leave_out))
X_out = X[-index_to_leave_out:]
y_out = y[-index_to_leave_out:]

X_in = X[:-index_to_leave_out]
y_in = y[:-index_to_leave_out]

clf = SVC(decision_function_shape='ovo', random_state=42)
scores = cross_val_score(clf, X_in, y_in, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))