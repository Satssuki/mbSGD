from csv import reader
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

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


def normalize_data(data):
    data_minmax = dataset_minmax(data)
    normalize_dataset(data, data_minmax)
    return data


def decision_data():
    file_name = 'C:\Users\cfilip09\Downloads\\binary.csv'
    data = load_csv(file_name)

    for ii in range(len(data[0])):
        str_column_to_float(data, ii)

    data = normalize_data(data)

    data = np.array(data)
    return data

# merge features from labes
def merge_labels_features(data_x, data_y):
    data = np.c_[data_x, data_y]
    return data

# generate a 2-class classification problem with 400 data points,
# where each data point is a 2D feature vector
def random_data():
    (X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)
    data = merge_labels_features(X, y)
    data = normalize_data(data)
    return data

# decision_data()
