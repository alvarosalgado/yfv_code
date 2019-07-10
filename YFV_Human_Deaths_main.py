#!/usr/bin/env python3
"""
From 'YFV_Ct_Callithrix_main_rev1.ipynb'
"""


from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Alphabet import IUPAC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("whitegrid")

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc


import shap

"""
#######################################################################
Import data
Import `.pkl` file that was created on "data_preprocessing_YFV.ipynb"
#######################################################################
"""
def get_data(pickle_seqdf, pickle_ohe):
    ohe_df = pd.read_pickle(pickle_ohe)
    seq_df = pd.read_pickle(pickle_seqdf)

    # Select only callithrix samples
    ohe_df_calli = ohe_df[ohe_df['Host'] == 'Callithrix']
    ohe_df_calli = ohe_df_calli.sort_values(by='Ct_Group')

    # Select alouatta samples for comparison later
    ohe_df_alou = ohe_df[ohe_df['Host'] == 'Alouatta']
    ohe_df_alou = ohe_df_alou.sort_values(by='Ct_Group')

    return (seq_df, ohe_df, ohe_df_calli, ohe_df_alou)

"""
#######################################################################
Train and test splits

Separate data into train and test sets.
Since the dataset is small and imbalanced, I will separate only 10% for testing.
#######################################################################
"""
def get_train_test_split(ohe_df_calli, test_size=0.1):
    # Get only the ohe nucleotide info in X
    X = ohe_df_calli.drop(["ID","Host","Ct","Date","Season","Ct_Group"], axis=1)
    # The target class is Ct_Group (high or low)
    y = ohe_df_calli["Ct_Group"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=0,
                                                        shuffle=True,
                                                        stratify=y)

    ## Dealing with imbalanced data
    scale_pos_weight = len(y_train)/y_train.sum()

    return (X, y, X_train, X_test, y_train, y_test, scale_pos_weight)

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
def initial_xgb_model(X_train, y_train, X_test, y_test, scale_pos_weight):
    # Initial XGB model
    initial_xgb = XGBClassifier(learning_rate=0.001,
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

    fig1, axes1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
    axes1[0].plot(results['validation_0']['error'], label='Train Error')
    axes1[0].plot(results['validation_1']['error'], label='Validation Error')
    axes1[0].set_title("Initial XGBoost Error")
    axes1[0].set_xlabel("Iteration")
    axes1[0].set_ylabel("Error")
    axes1[0].legend()

    axes1[1].plot(results['validation_0']['auc'], label='Train AUC-ROC')
    axes1[1].plot(results['validation_1']['auc'], label='Validation AUC-ROC')
    axes1[1].set_title("Initial XGBoost AUC-ROC")
    axes1[1].set_xlabel("Iteration")
    axes1[1].set_ylabel("AUC")
    axes1[1].legend()

    fig1.tight_layout();

    fig1.savefig('./figures/initial_xgb_model.png', format='png', dpi=300, transparent=False)

    return initial_xgb
"""
#######################################################################
Grid Search XGBoost
#######################################################################
"""
def grid_cv_xgb(X_train, y_train, scale_pos_weight, params, folds = 5):
    xgb = XGBClassifier(objective='binary:logistic', njobs=4, random_state=0, scale_pos_weight=scale_pos_weight)

    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    grid = GridSearchCV(estimator=xgb,
                        param_grid=params,
                        scoring='roc_auc',
                        n_jobs=4,
                        cv=skf.split(X_train,y_train),
                        verbose=3 )

    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    results = pd.DataFrame(grid.cv_results_)
    results.to_csv('xgb-grid-search-results-01.csv', index=False)

    return (best_params, results)
"""
#######################################################################
Final XGBoost model
Fitting the final XGBoost with parameters found on grid_cv.
Use all training data.
Test on test data.
#######################################################################
"""
def final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, best_params):

    xgb = XGBClassifier(**best_params)
    xgb.set_params(njobs=4,
                   random_state=0,
                   objective='binary:logistic',
                   scale_pos_weight=scale_pos_weight)

    eval_set = [(X_train, y_train), (X_test, y_test)]
    eval_metric = ["error", "auc"]

    xgb.fit(X_train, y_train,
            eval_metric=eval_metric,
            eval_set=eval_set)

    results = xgb.evals_result()

    fig1, axes1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
    axes1[0].plot(results['validation_0']['error'], label='Train Error')
    axes1[0].plot(results['validation_1']['error'], label='Validation Error')
    axes1[0].set_title("Final XGBoost Error")
    axes1[0].set_xlabel("Iteration")
    axes1[0].set_ylabel("Error")
    axes1[0].legend()

    axes1[1].plot(results['validation_0']['auc'], label='Train AUC-ROC')
    axes1[1].plot(results['validation_1']['auc'], label='Validation AUC-ROC')
    axes1[1].set_title("Final XGBoost AUC-ROC")
    axes1[1].set_xlabel("Iteration")
    axes1[1].set_ylabel("AUC")
    axes1[1].legend()

    fig1.tight_layout();

    fig1.savefig('./figures/final_xgb_model.png', format='png', dpi=300, transparent=False)

    return xgb

"""
#######################################################################
Random Forest
Based on experience, we will use a random forest with 100 trees
(we compared `oob_score_` values for different numbers of trees).

Set `random state = 0` and `oob_score = True` to allow reproducibility
and to use "out of bag" samples to compute accuracy.
#######################################################################
"""
def random_forest(X_train, y_train, X_test, y_test, n=100):

    rf = RandomForestClassifier(n_estimators=n,
                                random_state=0,
                                oob_score=True)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)
    return (rf, roc_auc, rf.oob_score_)

"""
#######################################################################
Modified Logistic Regression
#######################################################################
"""
import Logistic_regression_modified as lr

(e, ave_MSE, all_alphas, ave_alphas) = lr.LR_k_fold_CV(X_train, y_train, k=10)

normalized_alphas = lr.normalize_alphas(ave_alphas, X)

ave_acc = 1 - ave_MSE

fig_lr, axlr = plt.subplots(figsize=(10, 6))
axlr.scatter(range(df_alphas_std.shape[0]), df_alphas_std)
axlr.set_title("Logistic Regression Coeficients")
axlr.set_xlabel("Feature position (nucleotide position)")
axlr.set_ylabel("Coeficient value")

axlr.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

fig_lr.tight_layout();

fig_lr.savefig('./figures/lr_coef.png', format='png', dpi=300, transparent=False)

"""
#######################################################################
Feature Importances
#######################################################################
"""
# load JS visualization code to notebook
shap.initjs()

def get_explainer(xgb, rf, X_train):
    # Creates the explainer based on the model.
    rf_explainer = shap.TreeExplainer(rf, data=X_train)
    rf_shap_values = rf_explainer.shap_values(X_train)
    rf_shap_values = rf_shap_values[0] # For the random forest model, the shap TreeExplainer returns 2 sets of values, one for class 1 and one for class 0. They are symmetric, so you can use either.

    # Creates the explainer based on the model.
    xgb_explainer = shap.TreeExplainer(xgb, data=X_train)
    xgb_shap_values = xgb_explainer.shap_values(X_train)

    fig, axes = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
    for i in range(xgb_shap_values.shape[0]):
        axes[0].plot(xgb_shap_values[i])
    axes[0].set_title("XGBoost Feature Importances")
    axes[0].set_xlabel("Feature (nucleotide)")
    axes[0].set_ylabel("Importance (SHAP value)")
    axes[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    for i in range(rf_shap_values.shape[0]):
        axes[1].plot(rf_shap_values[i])
    axes[1].set_title("Random Forest Feature Importances")
    axes[1].set_xlabel("Feature (nucleotide)")
    axes[1].set_ylabel("Importance (SHAP value)")
    axes[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off


    fig.tight_layout();

    fig.savefig('./figures/importances_1.png', format='png', dpi=300, transparent=False)

    return (xgb_explainer, rf_explainer, xgb_shap_values, rf_shap_values)


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
def get_sorted_importances(shap_values_df):
    abs = np.abs(shap_values_df)
    abs_sum = np.sum(abs, axis=0)
    abs_sum_sort = abs_sum.sort_values(ascending=False)

    return abs_sum_sort



# nucleotides_df.index == xgb_shap_values_df.index



"""
#######################################################################
Summary Plots
#######################################################################
"""
fig_xgb_shap = plt.figure()
shap.summary_plot(xgb_shap_values_df.values,
                        nucleotides_df,
                        plot_type="bar",
                        max_display=10,
                        sort=True,
                        title="title")
fig_xgb_shap.savefig("./figures/fig_xgb_shap.png", format='png', dpi=300, transparent=False)

fig_fr_shap = plt.figure()
shap.summary_plot(rf_shap_values_df.values,
                        nucleotides_df,
                        plot_type="bar",
                        max_display=10,
                        sort=True)
fig_fr_shap.savefig("./figures/fig_fr_shap.png", format='png', dpi=300, transparent=False)


"""
#######################################################################
MAIN
#######################################################################
"""


pickle_ohe = '../Callithrix_Analysis/DATA/!CLEAN/YFV_seq_ohe_df.pkl'
pickle_seqdf = '../Callithrix_Analysis/DATA/!CLEAN/YFV_seq_df.pkl'

(X, y, X_train, X_test, y_train, y_test, scale_pos_weight) = get_train_test_split(ohe_df_calli, test_size=0.1)


# A parameter grid for XGBoost
params = {
        'subsample': [1.0],
        'colsample_bytree': [0.3],
        'max_depth': [3, 5],
        'learning_rate': [1, 0.01, 0.001],
        'n_estimators': [250, 5000]
        }

(best_params, results) = grid_cv_xgb(X_train, y_train, scale_pos_weight, params, folds = 5)

xgb = final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, best_params)

(rf, roc_auc, rf.oob_score_) = random_forest(X_train, y_train, X_test, y_test, n=100)

(xgb_explainer, rf_explainer, xgb_shap_values, rf_shap_values) = get_explainer(xgb, rf, X_train)

rf_shap_values_df = pd.DataFrame(rf_shap_values,
                                index=X_train.index,
                                columns=X_train.columns)

xgb_shap_values_df = pd.DataFrame(xgb_shap_values,
                                 index=X_train.index,
                                 columns=X_train.columns)

rf_shap_values_df = ohe_inverse(rf_shap_values_df)
xgb_shap_values_df = ohe_inverse(xgb_shap_values_df)

xgb_summary = get_sorted_importances(xgb_shap_values_df)
xgb_summary

rf_summary = get_sorted_importances(rf_shap_values_df)
rf_summary

calli_df[['Ct_Group', int(xgb_summary.index[0])]]
alou_df[['Ct_Group', int(xgb_summary.index[0])]]
