from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import dataloader
import numpy as np

(X, y) = dataloader.seperate_labels_features(dataloader.wine_data())
X_out = X[-40:]
y_out = y[-40:]


X_in = X[:-40]
y_in = y[:-40]

X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.2, random_state=42)


scaler = MinMaxScaler()

trainX = scaler.fit_transform(X_train)

testX = scaler.transform(X_test)

def next_batch(data_x, data_y, batch_size):
	# loop over our dataset `X` in mini-batches of size `batchSize`
	for i in np.arange(0, data_x.shape[0], batch_size):
	    # yield a tuple of the current batched data and labels
	    yield (data_x[i:i + batch_size], data_y[i:i + batch_size])

def sigmoid_activation(x):
	# compute and return the sigmoid activation value for a
	# given input value
	return 1.0 / (1 + np.exp(-x))

trainX = np.c_[np.ones((trainX.shape[0])), trainX]
trainX[0]
np.array([ 1.        ,  0.27801553,  0.12103743])

import matplotlib.pyplot as plt

number_of_epochs = 2000
alpha=0.01
batchSize=50

W = np.random.uniform(size=(trainX.shape[1],))
lossHistory = []
for epoch in np.arange(0, number_of_epochs):
	# initialize the total loss for the epoch
	epochLoss = []
	for (batchX, batchY) in next_batch(trainX, y_train, batchSize):
	        preds = sigmoid_activation(batchX.dot(W))
	        error = preds - batchY
	        loss = np.sum(error ** 2)
	        epochLoss.append(loss)
	        gradient = batchX.T.dot(error) / batchX.shape[0]
	        W += -alpha * gradient
	lossHistory.append(np.average(epochLoss))


Y = (-W[0] - (W[1] * trainX)) / W[2]

plt.figure()
plt.scatter(trainX[:, 1], trainX[:, 2], marker="o", c=y_train)
plt.plot(trainX, Y, "r-")
fig = plt.figure()
plt.plot(np.arange(0, number_of_epochs), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

