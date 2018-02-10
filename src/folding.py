import dataloaders
import numpy as np
from sklearn.model_selection import KFold
import dataloader

number_of_folds = 10

X_in, y_in = dataloader.seperate_labels_features(dataloader.random_data(2, 2))
kf = KFold(n_splits=number_of_folds, random_state=42, shuffle=True)
kf.get_n_splits(X_in)

print(kf)

for train_index, test_index in kf.split(X_in):
    X_train, X_test = X_in[train_index], X_in[test_index]
    print 'X_train' + str(len(X_train)) + 'X_test' + str(len(X_test))
    y_train, y_test = y_in[train_index], y_in[test_index]