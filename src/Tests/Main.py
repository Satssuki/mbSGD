import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn import linear_model
from sklearn.kernel_approximation import RBFSampler
import itertools
from sklearn.preprocessing import PolynomialFeatures


# calculcate new features using polynomial transformation
def polynomial_transformation(X):

    # poly = PolynomialFeatures(2)
    # return poly.fit_transform(X)

    result = list()
    index_result = list()

    for record in X:
        record = X[0]
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
        
        print index_feat

        result.append(record_result)

    return np.array(result)


epochs = 100
learning_rate = 0.01
batch_size = 32

# generate a 2-class classification problem with 400 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=400, n_features=2,
                    centers=2, cluster_std=2.5, random_state=95)

# insert a column of 1's as the first entry in the feature
# vector -- this is a little trick that allows us to treat
# the bias as a trainable parameter *within* the weight matrix
# rather than an entirely separate variable
X = np.c_[np.ones((X.shape[0])), X]


rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)

dd = [[1 ,2 ,3, 5, 5]]

test = polynomial_transformation(dd)
print np.array(test)

poly = PolynomialFeatures(2)
print poly.fit_transform(dd)

# print X[0]
# print X_features[0]
