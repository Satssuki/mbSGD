from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import dataloader
import numpy as np
from numpy.linalg import norm
import pandas
from sklearn.model_selection import KFold


def next_batch(data_x, data_y, batch_size):
    for i in np.arange(0, data_x.shape[0], batch_size):
        yield (data_x[i:i + batch_size], data_y[i:i + batch_size])


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return sigmoid_activation(yhat)


def accuracy_metric(actual, predicted):
    correct = 0

    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) * 100.0


def shuffle_data(data_x, data_y):
    data = dataloader.merge_labels_features(data_x, data_y)
    np.random.shuffle(data)
    return dataloader.seperate_labels_features(data)


def train(X_train, X_test, y_train, y_test, number_of_epochs, alpha, batchSize, add_additional_feature):
    scaler = MinMaxScaler()
    trainX = scaler.fit_transform(X_train)
    testX = scaler.transform(X_test)

    # print 'before'
    # print trainX[0]

    if(add_additional_feature):
        trainX = np.c_[np.ones((trainX.shape[0])), trainX]

    # print 'after'
    # print trainX[0]

    W = np.random.uniform(size=(trainX.shape[1],))
    lossHistory = []
    for epoch in np.arange(0, number_of_epochs):
        epochLoss = []
        shuffled_data_x, shuffled_data_y = shuffle_data(trainX, y_train)
        for (batchX, batchY) in next_batch(shuffled_data_x, shuffled_data_y, batchSize):
            preds = sigmoid_activation(batchX.dot(W))
            error = preds - batchY
            loss = np.sum(error ** 2)
            epochLoss.append(loss)
            gradient = batchX.T.dot(error) / batchX.shape[0]
            W += -alpha * gradient
        lossHistory.append(np.average(epochLoss))

    # Y = ((W[0] * trainX)) / W[1]
    Y = (-W[0] - (W[1] * trainX)) / W[2]
    # Y = np.dot(trainX, W)

    predictions = list()

    for i in range(len(testX)):
        predicted = predict(testX[i], W)
        predicted = round(predicted)
        predictions.append(predicted)

    accuracy = accuracy_metric(y_test, predictions)

    return W, lossHistory, accuracy

    plt.figure()
    plt.scatter(trainX[:, 1], trainX[:, 2], marker="o", c=y_train)
    plt.plot(trainX, Y, "r-")
    fig = plt.figure()
    plt.plot(np.arange(0, number_of_epochs), lossHistory)
    fig.suptitle("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()
    return W, lossHistory, accuracy


def get_classes(labels):
    classes = list(labels)
    unique_classes = np.unique(classes)

    if len(unique_classes) == 2:
        return [classes]

    print unique_classes

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


def choose_the_best_model(unknown_data_X, unknown_data_y, coeffecients):
    model_accuracies = list()

    for i in range(len(coeffecients)):
        W = coeffecients[i]
        model_predictions = list()

        for i in range(len(unknown_data_X)):
            predicted = predict(unknown_data_X[i], W)
            predicted = round(predicted)
            model_predictions.append(predicted)

        model_accuracy = accuracy_metric(unknown_data_y, model_predictions)
        model_accuracies.append(model_accuracy)

    return coeffecients[np.argmax(model_accuracies)]


n_epochs = 2000
l_rate = 0.001
batch_size = 50
add_feature = True
L2_reg = 0.02
number_of_folds = 5
leave_out = 0.2
# X_in, y_in = dataloader.seperate_labels_features(dataloader.wine_data())
# X_in, y_in = dataloader.seperate_labels_features(dataloader.decision_data())
# X_in, y_in = dataloader.seperate_labels_features(dataloader.local_data("dataset42_1.csv"))
X, y = dataloader.seperate_labels_features(dataloader.random_data(2, 2))
index_to_leave_out = int(round(len(X) * leave_out))
print 'data length: ' + str(len(X))
print 'leaving out : ' + str(index_to_leave_out) + ' records (' + str(leave_out * 100) + '%)'

X_out = X[-index_to_leave_out:]
y_out = y[-index_to_leave_out:]

X_in = X[:-index_to_leave_out]
y_in = y[:-index_to_leave_out]


X_train, X_test, y_train, y_test = [], [], [], []

kf = KFold(n_splits=number_of_folds, random_state=42, shuffle=True)

accuracies = list()
coeffecients = list()

for train_index, test_index in kf.split(X_in):
    X_train, X_test = X_in[train_index], X_in[test_index]
    y_train, y_test = y_in[train_index], y_in[test_index]
    # X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.2, random_state=42)
    W, lossHistory, accuracy = train(X_train, X_test, y_train, y_test, n_epochs, l_rate, batch_size, add_feature)
    accuracies.append(accuracy)
    coeffecients.append(W)

for coef in coeffecients:
    print 'coeffecients: ' + str(coef)

print 'accuracies: ' + str(accuracies)
best_model = choose_the_best_model(X_out, y_out, coeffecients)
print 'best_model: ' + str(best_model)
print 'avg accuracy: ' + str(np.mean(accuracies))
