import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn import linear_model
import sgd
epochs = 100
learning_rate = 0.01
batch_size = 32

# generate a 2-class classification problem with 400 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)

# insert a column of 1's as the first entry in the feature
# vector -- this is a little trick that allows us to treat
# the bias as a trainable parameter *within* the weight matrix
# rather than an entirely separate variable
X = np.c_[np.ones((X.shape[0])), X]

# initialize our weight matrix such it has the same number of
# columns as our input features
print("[INFO] starting training...")
Y, lossHistory = sgd.mbSGD(epochs, learning_rate, batch_size, X, y)
print X[0]
# sgd.plot_data(X, Y, y, epochs, lossHistory)

# clf = linear_model.SGDRegressor(penalty="l2", eta0=learning_rate, max_iter=epochs)
# print clf.fit(X, y)
#
# print clf.get_params()
