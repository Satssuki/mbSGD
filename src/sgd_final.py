from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import dataloader
import numpy as np
from numpy.linalg import norm
import pandas

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

# Calculate accuracy percentage
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

number_of_epochs = 2000
alpha = 0.01
batchSize = 50
add_additional_feature = True
L2_reg = 0.02

# X_in, y_in = dataloader.seperate_labels_features(dataloader.wine_data())
# X_in, y_in = dataloader.seperate_labels_features(dataloader.decision_data())
# X_in, y_in = dataloader.seperate_labels_features(dataloader.local_data("dataset42_1.csv"))
X_in, y_in = dataloader.seperate_labels_features(dataloader.random_data(2, 2))

X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
trainX = scaler.fit_transform(X_train)
testX = scaler.transform(X_test)

print 'before'
print trainX[0]

if(add_additional_feature):
        trainX = np.c_[np.ones((trainX.shape[0])), trainX]

print 'after'
print trainX[0]

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

print 'coef: ' + str(W)
print 'trainX: '
print trainX

#Y = ((W[0] * trainX)) / W[1]
Y = (-W[0] - (W[1] * trainX)) / W[2]
#Y = np.dot(trainX, W)

predictions = list()

for i in range(len(testX)):
	predicted = predict(testX[i], W)
	predicted = round(predicted)
	predictions.append(predicted)

accuracy = accuracy_metric(y_test, predictions)

print 'lossHistory: ' + str(lossHistory)
print 'accuracy: ' + str(accuracy)

plt.figure()
plt.scatter(trainX[:, 1], trainX[:, 2], marker="o", c=y_train)
plt.plot(trainX, Y, "r-")
fig = plt.figure()
plt.plot(np.arange(0, number_of_epochs), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()