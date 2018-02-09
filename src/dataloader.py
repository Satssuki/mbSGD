from csv import reader
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd

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

def download_data(url,label_index_is_last=True):
    pandas_data = pd.read_csv(url)
    x, y = seperate_labels_features(pandas_data.values, label_index_is_last=label_index_is_last)
    data = merge_labels_features(x, y)
    return np.array(data)


def wine_data():
    return download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', True)


def diabetes_data():
    return download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data', False)


def decision_data():
    file_name = 'C:\PycharmProjects\mbSGD\data\\binary.csv'
    data = load_csv(file_name)

    for ii in range(len(data[0])):
        str_column_to_float(data, ii)

    x, y = seperate_labels_features(np.array(data), label_index_is_last=False)
    x = normalize_data(x)
    data = merge_labels_features(x, y)
    return np.array(data)

# generate a 2-class classification problem with 400 data points,
# where each data point is a 2D feature vector
def random_data():
    (X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)
    return merge_labels_features(X, y)

# merge features from labes
def merge_labels_features(data_x, data_y):
    data = np.c_[data_x, data_y]
    return data

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
