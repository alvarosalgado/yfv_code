#!/usr/bin/env python3
%matplotlib inline


from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Alphabet import IUPAC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc


import shap
shap.__version__

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

## Dealing with imbalanced data
scale_pos_weight = len(y_train)/y_train.sum()

"""
#######################################################################
Classifiers
#######################################################################
"""

"""
#######################################################################
XGBoost parameter tuning
#######################################################################
"""
# Initial XGB model
xgb = XGBClassifier(learning_rate=0.001,
                    colsample_bytree = 0.3,
                    subsample = 1,
                    objective='binary:logistic',
                    n_estimators=10000,
                    max_depth=3,
                    njobs=4,
                    random_state=0,
                    scale_pos_weight=scale_pos_weight
                    )

eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["error", "auc"]

xgb.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set)

results = xgb.evals_result()

fig1, ax1 = plt.subplots()
ax1.plot(results['validation_0']['error'], label='Train Error')
ax1.plot(results['validation_1']['error'], label='Validation Error')
ax1.legend();

fig2, ax2 = plt.subplots()
ax2.plot(results['validation_0']['auc'], label='Train AUC-ROC')
ax2.plot(results['validation_1']['auc'], label='Validation AUC-ROC')
ax2.legend();

y_total_pred = xgb.predict(X)
print(y_total_pred)
print(y.values)

"""
#######################################################################
Grid Search XGBoost
#######################################################################
"""

xgb = XGBClassifier(objective='binary:logistic', njobs=4, random_state=0, scale_pos_weight=scale_pos_weight)

# A parameter grid for XGBoost
params = {
        'subsample': [1.0],
        'colsample_bytree': [0.3],
        'max_depth': [3, 5],
        'learning_rate': [1, 0.01, 0.001],
        'n_estimators': [250, 5000]
        }

folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

grid = GridSearchCV(estimator=xgb,
                    param_grid=params,
                    scoring='roc_auc',
                    n_jobs=4,
                    cv=skf.split(X_train,y_train),
                    verbose=3 )

grid.fit(X_train, y_train)

best_params = grid.best_params_
print("Best Parameters: \n", best_params)
results = pd.DataFrame(grid.cv_results_)
results.to_csv('xgb-grid-search-results-01.csv', index=False)

results
"""
#######################################################################
Final XGBoost model
Fitting the final XGBoost with parameters found on grid_cv.
Use all training data.
Test on test data.
#######################################################################
"""
params = best_params
# params = {'colsample_bytree': 0.6,
#           'learning_rate': 0.01,
#           'max_depth': 3,
#           'n_estimators': 250,
#           'subsample': 1.0}

xgb = XGBClassifier(**params)
xgb.set_params(silent=True,
               verbosity=0,
               njobs=4,
               random_state=0,
               objective='binary:logistic',
               scale_pos_weight=scale_pos_weight)

eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["error", "auc"]

xgb.fit(X_train, y_train,
        eval_metric=eval_metric,
        eval_set=eval_set,
        verbose=False)

results = xgb.evals_result()

fig1, ax1 = plt.subplots()
ax1.plot(results['validation_0']['error'], label='Train Error')
ax1.plot(results['validation_1']['error'], label='Test Error')
ax1.set_xlabel("iteration")
ax1.set_ylabel("error")
ax1.set_title("XGBoost train and test error")
ax1.legend();

fig2, ax2 = plt.subplots()
ax2.plot(results['validation_0']['auc'], label='Train AUC-ROC')
ax2.plot(results['validation_1']['auc'], label='Test AUC-ROC')
ax2.set_xlabel("iteration")
ax2.set_ylabel("AUC-ROC")
ax2.set_title("XGBoost train and test AUC-ROC")
ax2.legend();


y_total_pred = xgb.predict(X)
print(y_total_pred)
print(y.values)

error = y_total_pred - y.values

print(np.abs(error))
"""
#######################################################################
Random Forest
Based on experience, we will use a random forest with 100 trees
(we comparer `oob_score_` values for different numbers of trees).

Set `random state = 0` and `oob_score = True` to allow reproducibility
and to use "out of bag" samples to compute accuracy.
#######################################################################
"""

rf = RandomForestClassifier(n_estimators=100,
                            random_state=0,
                            oob_score=True)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)
print(rf.oob_score_)

y_total_pred = rf.predict(X)
print(y_total_pred)
print(y.values)

error = y_total_pred - y.values

print(np.abs(error))

"""
#######################################################################
Modified Logistic Regression
#######################################################################
"""

"""
#######################################################################
Feature Importances
#######################################################################
"""
# load JS visualization code to notebook
shap.initjs()

# Creates the explainer based on the model.
rf_explainer = shap.TreeExplainer(rf, data=X)
rf_shap_values = rf_explainer.shap_values(X)
rf_shap_values = rf_shap_values[0] # For the random forest model, the shap TreeExplainer returns 2 sets of values, one for class 1 and one for class 0. They are symmetric, so you can use either.

# Creates the explainer based on the model.
xgb_explainer = shap.TreeExplainer(xgb, data=X)
xgb_shap_values = xgb_explainer.shap_values(X)

# for i in range(rf_shap_values.shape[0]):
#     plt.plot(rf_shap_values[i])
#
# for i in range(xgb_shap_values.shape[0]):
#     plt.plot(xgb_shap_values[i])

"""
#######################################################################
From one-hot encoding to original features' names.

Since "ohe" creates additional columns to accomodate the attributes'
categories, we have to go back to the original attribute's names
in order to clearly analyse the results, especially where it concerns
feature importances.

Procedure
To do so, we will sum the shap values of all categories of each attribute,
for each sample, as suggested by SHAP's developer
([see here](https://github.com/slundberg/shap/issues/397))
#######################################################################
"""
rf_shap_values_df = pd.DataFrame(rf_shap_values,
                                index=X.index,
                                columns=X.columns)

xgb_shap_values_df = pd.DataFrame(xgb_shap_values,
                                 index=X.index,
                                 columns=X.columns)

"""
#######################################################################
Inverse OHE function specifically for SHAP values
#######################################################################
"""
def ohe_inverse(df_shap_values):
    """Converts a dataframe containing shap values in ohe format
    back to original genomic positions"""

    # Auxiliary list to recreate original shap_values dataframe
    list_shap_original = []

    # Regular expression to pick attributes names.
    # Since in our case attributes names are the genomic positions (i.e. an integer number), we use the regex below
    import re
    pattern = "^\d+"

    # Auxiliary dictionary to create one pd.DataFrame for each sample, summing the shap values for each attribute.
    # Later, these dataframes will be appended together, resulting in the final df.
    dic={}

    # for each sample.
    for i, sample in df_shap_values.iterrows():
        # initialize an empty dictionary, that will contain "attribute : summed shap values" for
        # all attributes in this sample.
        dic = {}
        # The code below sums the importances for each category in each attribute in this sample.
        for pos in sample.index:
            attr = re.match(pattern, pos).group()
            if attr not in dic.keys():
                dic[attr] = sample[pos]
            else:
                dic[attr] += sample[pos]
        # Create a df containing only the current sample
        df_sample = pd.DataFrame(dic, index=[i])
        # Append it to a list that will become the full dataframe later
        list_shap_original.append(df_sample)

    # Create a DataFrame containing the shap values for the "original" attributes.
    shap_original = pd.concat(list_shap_original, axis=0)
    return shap_original
"""
#######################################################################
#######################################################################
"""

rf_shap_values_df = ohe_inverse(rf_shap_values_df)
xgb_shap_values_df = ohe_inverse(xgb_shap_values_df)

# rf_shap_values_df.shape
# xgb_shap_values_df.shape

nucleotides_df = seq_df[seq_df['Host'] == 'Callithrix']
nucleotides_df = nucleotides_df.drop(["ID","Host","Ct","Date", "Season", "Ct_Group"], axis=1);

# nucleotides_df.index == xgb_shap_values_df.index

fig1, ax1 = plt.subplots()
ax1 = shap.summary_plot(xgb_shap_values_df.values,
                        nucleotides_df,
                        plot_type="bar",
                        max_display=10,
                        sort=True)

fig2, ax2 = plt.subplots()
ax2 = shap.summary_plot(rf_shap_values_df,
                        nucleotides_df,
                        plot_type="bar",
                        max_display=30,
                        sort=True)

numeric_nucleotides_df = nucleotides_df.replace("A", 0.0)
numeric_nucleotides_df = numeric_nucleotides_df.replace("C", 1.0)
numeric_nucleotides_df = numeric_nucleotides_df.replace("T", 2.0)
numeric_nucleotides_df = numeric_nucleotides_df.replace("G", 3.0)
numeric_nucleotides_df = numeric_nucleotides_df.replace(np.nan, -1.0)

"""
#######################################################################
Summary Plots
#######################################################################
"""
# shap.summary_plot(xgb_shap_values_df.values, numeric_nucleotides_df, plot_type='dot', alpha=0.3)
# shap.summary_plot(rf_shap_values_df.values, numeric_nucleotides_df)

"""
#######################################################################
SHAP interaction values
#######################################################################
"""

def interactions(model_explainer, dataset, type="xgb"):
    """Had to change to know the type of model to work for RF, because their
    SHAP values are composed of two sets,
    one for each class, I don't know why"""

    shap_interaction_values = model_explainer.shap_interaction_values(dataset)
    if type == "rf":
        shap_interaction_values = shap_interaction_values[0]

    interaction_matrix = np.abs(shap_interaction_values[0])
    for i in range(1, shap_interaction_values.shape[0]):
        new_matrix = np.abs(shap_interaction_values[i])
        interaction_matrix = np.add(interaction_matrix, new_matrix)

    interaction_matrix = pd.DataFrame(interaction_matrix, index=dataset.columns, columns=dataset.columns)

    # The matrix containing only the off-diagonal values, i.e., the interaction values
    off_diag = interaction_matrix - np.diag(np.diagonal(interaction_matrix))

    interact_dic = {}
    for col in off_diag.columns:
        if len(off_diag.index[off_diag[col]>0].tolist()):
            interact_dic[col] = off_diag.index[off_diag[col]>0].tolist()


    # Now, I need to rank the largest interaction values and see where they happen.
    # To do so, I will flatten the off_diag array, sort its values,
    # use np.where to find them in the original 2d array, and get the indexes.
    # flat = off_diag.flatten()
    # flat = np.sort(flat)[::-1]
    #
    # # I will go through the values, skipping alternately because the interaction
    # # matrix is symmetric. Stop when values reach zero.
    # dic = {}
    # columns = X.columns
    # for i in range(0, len(flat)+1, 2):
    #     if flat[i] <= 0:
    #         break
    #     indexes = np.where(off_diag == flat[i])
    #     indexes = str((indexes[0][0], indexes[1][0]))
    #     value = flat[i]
    #     dic[indexes] = value
    #
    # high_interactions = pd.Series(dic)
    # high_interactions.sort_values(ascending=True, inplace=True)

    return (shap_interaction_values, interaction_matrix, off_diag, interact_dic)

"""
#######################################################################
#######################################################################
"""


(xgb_shap_interaction_values, xgb_interaction_matrix, xgb_off_diag, interact_dict) = interactions(xgb_explainer, X)
#
# plt.barh(xgb_high_interactions.index, xgb_high_interactions)

interact_dict

abs = np.abs(xgb_shap_values_df)
sum_abs = abs.sum(axis=0)
sorted_sum_abs = sum_abs.sort_values(ascending=False)

xgb_summary = sorted_sum_abs
# xgb_summary_30 = xgb_summary[:30]

abs = np.abs(rf_shap_values_df)
sum_abs = abs.sum(axis=0)
sorted_sum_abs = sum_abs.sort_values(ascending=False)

rf_summary = sorted_sum_abs


calli_df = seq_df[seq_df['Host']=='Callithrix']
calli_df = calli_df.sort_values(by='Ct_Group')

alou_df = seq_df[seq_df['Host']=='Alouatta']
alou_df = alou_df.sort_values(by='Ct_Group')

calli_df[['Ct_Group', int(xgb_summary.index[0])]]
alou_df[['Ct_Group', 8918]]

calli_df[['Ct_Group', int(rf_summary.index[1])]]

'''
fig1.savefig('Scatter_seasons.png', format='png', dpi=300, transparent=False)
'''
