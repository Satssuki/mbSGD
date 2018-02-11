import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn import linear_model
from sklearn.kernel_approximation import RBFSampler
import itertools


def get_polynomial_fetures(X):
    final_f = list()
    X1_features = itertools.combinations_with_replacement(np.arange(X), X) 

    for i in X1_features:
        final_f.append(np.array(i[0: (X)]))

    return final_f

def polynomial_all_fetures(X):
    number_of_features = len(X[0])
    final_f = get_polynomial_fetures(number_of_features)
    result = list()

    for record in X:
        record = X[0]
        record_result = list()
        record_result.append(1)

        for f in range(number_of_features):
            record_result.append(record[f])

        for expression in final_f:
            r_value = 1
            for i in range(number_of_features):
                r_value = r_value * record[expression[i]]
            
            print 'final_f'
            print expression
            print record
            print r_value
            record_result.append(r_value)

    result.append(record_result)
    return result
        
    



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



rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)


final_f = list()
X1_features = itertools.combinations_with_replacement([1, 2], 2) 

test = polynomial_all_fetures([[1, 2, 3]])

print test

# print X[0]
# print X_features[0]