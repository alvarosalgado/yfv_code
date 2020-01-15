#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import LeaveOneOut
X = np.array([[1, 2, 1, 2, 2, 3, 5], [1, 2, 1, 2, 2, 3, 5],[1, 2, 1, 2, 2, 3, 5],[1, 2, 1, 2, 2, 3, 5],[1, 2, 1, 2, 2, 3, 5],[1, 2, 1, 2, 2, 3, 5]])
y = np.ones((X.shape[0]))
loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
  print('Train index: {}'.format(train_index))
  print('Test index: {}'.format(test_index))



for train_index, test_index in loo.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
