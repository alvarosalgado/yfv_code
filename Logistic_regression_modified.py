#!/usr/bin/env python3

""" Modified Logistic Regression

The code below was adapted from the work of Francielly Morais-Rodrigues, "Analysis of the genes important for breast cancer progression after the application modified logistic regression".

Notebook author: Álvaro Salgado
salgado.alvaro@me.com

"""

%matplotlib inline

from Bio.Seq import Seq
from Bio import SeqIO
from Bio import SeqFeature
from Bio.Alphabet import IUPAC

import pandas as pd
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit # generates k-fold by randomly sampling dataset
from sklearn.model_selection import StratifiedKFold # generates k-fold keeping ratio of classes in each fold
from sklearn.model_selection import StratifiedShuffleSplit # generates k-fold by randomly sampling dataset keeping ratio of classes in each fold

import re

import shap

"""
#######################################################################

#######################################################################
"""

def logistic_regression(X, y):
    '''Performs "Modified Logistic Regression" on the dataset provided'
    X = (m x n+1) pd.DataFrame
    y = "class" pd.Series and contain the sample's class codification (0 and 1).
    If data is categorical (our case), dataset must be provided in "one-hot encoding" format (see pd.get_dummies()).
    Returns alpha vector containing adjusted parameters.
    '''

    # Auxiliary variables to deal with numeric convergence issues.
    c1 = np.log(0.99/(1-0.99))
    c0 = np.log(0.01/(1-0.01))

    # normalizes type
    X_ohe = X.astype(int)
    y = y.astype(int)

    A = X_ohe
    b = np.zeros(len(y))
    b0 = y

    # matrix dimensions
    m = A.shape[0]
    n = A.shape[1]+1

    # Transforms values in vector "b" to avoid numerical problems when using log or exp.
    for i in range(m):
        if b0[i] == 1:
            b[i] = c1
        else:
            b[i] = c0

    B = A
    B.insert(loc=0, column='alpha_0', value=1)

    B = B.values
    I = np.eye(n)

    # (I + B^T B)*alpha = B^T b

    # left = (I + B^T B)
    # right = (B^T b)

    left = I + np.dot(B.T, B)
    right = np.dot(B.T, b)
    alphas = lin.solve(left, right)

    return (alphas)

"""
#######################################################################

#######################################################################
"""

def LR_predict(alphas, X):
    '''Given a dataset X(m,n) and alphas vector (n+1,1)
    Returns predictions (probabilities) for each
    of the "m" samples.
    P > 0.5, class 1
    P < 0.5, class 0
    '''
    X.insert(loc=0, column='alpha_0', value=1)

    X = X.values
    log_odds = np.dot(X, alphas)
    odds = np.exp(log_odds)
    P = (odds/(1+odds))
    predicted = np.where(P>0.5, 1, 0)

    return (P, predicted)

"""
#######################################################################
 - Average alphas (avoiding over-fitting in attribute's importances)
Since each run in the k-fold CV generates a different set of alphas depending on which samples were used for training, and because we are interested in understanding the dataset's information regarding its attribute's importances, it is a good policy to avoid over-fitting.

Therefore, we will use the average value of alphas from all the "k" runs to determine attribute importances.
#######################################################################
"""

def LR_k_fold_CV(X, y, k=10):
    '''Performs k-fold CV on dataset.
    X and y are pd.DataFrame and pd.Series
    Returns MSE for each k-fold.
    Returns average MSE
    '''
    # Test Size
    t_s = 1/k

    # normalizes type
    X = X.astype(int)
    y = y.astype(int)

    # save errors
    e = np.zeros(k)

    # save alphas for each run, to compute average alphas,
    # for attribute importance purposes.
    m = X.shape[0]
    n = X.shape[1]
    all_alphas = np.empty((n+1, k)) # alpha vectors are the columns. One vector for each run.

    # create splitter
    sss = StratifiedShuffleSplit(n_splits=k, test_size=t_s, random_state=0)

    i = 0
    for train_index, test_index in sss.split(X, y):
        print("fold k{0}, ".format(i))
        X_train = X.iloc[train_index, :]
        X_test  = X.iloc[test_index, :]

        y_train = y.iloc[train_index]
        y_test  = y.iloc[test_index]

        alphas = logistic_regression(X_train, y_train)
        all_alphas[:,i] = alphas

        P, predicted = predict(alphas, X_test)

        # MSE = 1/m * sum((y_hat - y)^2)
        squared_error = (y_test - predicted)**2
        m = len(y_test)

        MSE = (1/m) * squared_error.sum()

        e[i] = MSE

        i += 1

    ave_MSE = 1/len(e) * e.sum()
    ave_alphas = all_alphas.sum(axis=1)/k

    print("Finished!")
    return (e, ave_MSE, all_alphas, ave_alphas)

"""
#######################################################################
 Normalize and sort average alphas
 # Standardize alphas for feature importance
 see ["How to find the importance of the features for a logistic regression model?
 "](https://stackoverflow.com/questions/34052115/how-to-find-the-importance-of-the-features-for-a-logistic-regression-model)
#######################################################################
"""
def normalize_alphas(ave_alphas, X):
    df_alphas = pd.Series(ave_alphas)
    df_alphas = df_alphas.drop(0)
    df_alphas.index = X.columns

    df_alphas_std = np.std(X, axis = 0) * df_alphas

    df_alphas_std.sort_values(ascending=False, inplace=True)

    return df_alphas_std

"""
#######################################################################
 MAIN
#######################################################################
"""

"""
#######################################################################
Import data
Import `.pkl` file that was created on "data_preprocessing_YFV.ipynb"
#######################################################################
"""
ohe_df = pd.read_pickle('../DATA/!CLEAN/YFV_seq_ohe_df.pkl')
seq_df = pd.read_pickle('../DATA/!CLEAN/YFV_seq_df.pkl')

ohe_df_calli = ohe_df[ohe_df['Host'] == 'Callithrix']
ohe_df_calli = ohe_df_calli.sort_values(by='Ct_Group')

ohe_df_alou = ohe_df[ohe_df['Host'] == 'Alouatta']

"""
#######################################################################
Train and test splits
#######################################################################
"""
X = ohe_df_calli.drop(["ID","Host","Ct","Date","Season","Ct_Group"], axis=1)
y = ohe_df_calli["Ct_Group"]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=0,
                                                    shuffle=True,
                                                    stratify=y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

"""
#######################################################################
Fit Logistic Regression
#######################################################################
"""

e, ave_MSE, all_alphas, ave_alphas = LR_k_fold_CV(X_train, y_train)

ave_acc = 1 - ave_MSE
print("average accuracy =", ave_acc)

for error in e:
    print(1 - error)

plt.scatter(range(ave_alphas.shape[0]), ave_alphas);

# Create a pd.Series from average alphas.
# Index it by attribute name, from X.columns.

df_alphas_std = normalize_alphas(ave_alphas, X)

plt.scatter(range(df_alphas_std.shape[0]), df_alphas_std);


df_alphas_std[:10]
df_alphas_std[-10:]
