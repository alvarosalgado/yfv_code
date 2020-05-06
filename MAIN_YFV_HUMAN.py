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

import sklearn
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.utils.multiclass import unique_labels

import pickle
import datetime
import re
import glob
import time
import progressbar

import os, sys
cwd = os.getcwd()

import shap

# import xgboost
# xgboost.__version__
# sklearn.__version__
"""
#######################################################################
Import data
Import `.pkl` file that was created on "data_preprocessing_YFV.ipynb"
#######################################################################
"""
def get_data(pickle_seqdf, pickle_ohe, pickle_seqdforiginal):
    ohe_df = pd.read_pickle(pickle_ohe)
    seq_df = pd.read_pickle(pickle_seqdf)
    seq_df_original = pd.read_pickle(pickle_seqdforiginal)

    return (seq_df, ohe_df, seq_df_original)

"""
#######################################################################
Train and test splits

Separate data into train and test sets.
Since the dataset is small and imbalanced, I will separate only 10% for testing.
#######################################################################
"""
def get_train_test_split(ohe_df, test_size=0.1):
    # Get only the ohe nucleotide info in X
    X = ohe_df.drop(["Library", "BC", "ID", "Host", "Class", "Dataset"], axis=1)
    # The target class is Ct_Group (high or low)
    y = ohe_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=0,
                                                        shuffle=True,
                                                        stratify=y)

    ## Dealing with imbalanced data
    # scale_pos_weight = sum(negative instances) / sum(positive instances)
    # source: https://xgboost.readthedocs.io/en/latest/parameter.html
    positive = y_train.sum()
    negative = len(y_train) - positive
    scale_pos_weight = negative/positive
    # scale_pos_weight = len(y_train)/y_train.sum()

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
def initial_xgb_model(X_train, y_train, X_test, y_test, scale_pos_weight, analysis):
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

    fig1, axes1 = plt.subplots(figsize=(10, 8), nrows=1, ncols=2)
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

    fig1.savefig(fig_dir+'/{}_XGB_initial.png'.format(analysis), format='png', dpi=300, transparent=False)

    return initial_xgb
"""
#######################################################################
Grid Search XGBoost
#######################################################################
"""

# sorted(sklearn.metrics.SCORERS.keys())

def grid_cv_xgb(X_train, y_train, scale_pos_weight, params, analysis, folds = 5):
    xgb = XGBClassifier(objective='binary:logistic', njobs=4, random_state=0, scale_pos_weight=scale_pos_weight)

    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    grid = GridSearchCV(estimator=xgb,
                        param_grid=params,
                        scoring=['balanced_accuracy', 'roc_auc'],
                        return_train_score=True,
                        n_jobs=4,
                        cv=skf.split(X_train,y_train),
                        verbose=1,
                        refit='roc_auc')

    grid.fit(X_train, y_train)

    # best_params = grid.best_params_
    # results = pd.DataFrame(grid.cv_results_)
    # results.to_csv('{}_xgb-grid-search-results-01.csv'.format(analysis), index=False)

    return (grid)
"""
#######################################################################
Final XGBoost model
Fitting the final XGBoost with parameters found on grid_cv.
Use all training data.
Test on test data.
#######################################################################
"""
def final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, best_params, analysis):

    xgb = XGBClassifier(**best_params)
    xgb.set_params(njobs=4,
                   random_state=0,
                   objective='binary:logistic',
                   scale_pos_weight=scale_pos_weight)

    eval_set = [(X_train, y_train), (X_test, y_test)]
    eval_metric = ["error", "auc"]

    xgb.fit(X_train, y_train,
            eval_metric=eval_metric,
            eval_set=eval_set,
            verbose=0)

    results = xgb.evals_result()

    fig1, axes1 = plt.subplots(figsize=(10, 8), nrows=1, ncols=2)
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

    fig1.savefig(fig_dir+'/{}_final_xgb_model.png'.format(analysis), format='png', dpi=300, transparent=False)

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
# def random_forest(X_train, y_train, X_test, y_test, cw, n=100):
#
#     rf = RandomForestClassifier(n_estimators=n,
#                                 random_state=0,
#                                 max_features = 'auto',
#                                 bootstrap=True,
#                                 oob_score=True,
#                                 class_weight={0:1, 1:cw})
#
#     rf.fit(X_train, y_train)
#
#     y_pred_prob = rf.predict_proba(X_test)
#
#     fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
#
#     roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
#     score = rf.score(X_test, y_test)
#
#     return (rf, roc_auc, rf.oob_score_, score, fpr, tpr)


"""
#######################################################################
Modified Logistic Regression
#######################################################################
"""
def logistic_regression(X_train, y_train, X, k=10):
    import Logistic_regression_modified as lr

    (e, ave_MSE, all_alphas, ave_alphas) = lr.LR_k_fold_CV(X_train, y_train, k)

    ohe_normalized_alphas = lr.normalize_alphas(ave_alphas, X)
    alphas = ohe_inverse_LR(ohe_normalized_alphas)

    #  Use ohe_normalized_alphas to predict
    return (alphas, ohe_normalized_alphas, e, ave_MSE)

"""
#######################################################################
Plot ROC
#######################################################################
"""
def plot_roc(fpr, tpr, roc_auc, analysis, method, dataset='test_dataset'):
    fig, axes = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    lw = 2
    axes.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area ={0:.2f})'.format(roc_auc))
    axes.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')
    axes.set_title('Receiver operating characteristic - {}'.format(dataset))
    axes.legend(loc="lower right")
    fig.show()
    fig.savefig(fig_dir+"/{0}_ROC_{1}_{2}.png".format(analysis, method, dataset), format='png', dpi=300, transparent=False)

"""
#######################################################################
Feature Importances
#######################################################################
"""
# load JS visualization code to notebook
shap.initjs()

def get_explainer(xgb, rf, X_explain):
    # Creates the explainer based on the model.
    rf_explainer = shap.TreeExplainer(rf, data=None)
    rf_shap_values = rf_explainer.shap_values(X_explain)
    rf_shap_values = rf_shap_values[0] # For the random forest model, the shap TreeExplainer returns 2 sets of values, one for class 1 and one for class 0. They are symmetric, so you can use either.

    # Creates the explainer based on the model.
    xgb_explainer = shap.TreeExplainer(xgb, data=None)
    xgb_shap_values = xgb_explainer.shap_values(X_explain)



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
def ohe_inverse_LR(normalized_alphas):
    """Converts a dataframe containing shap values in ohe format
    back to original genomic positions"""

    normalized_alphas = np.abs(normalized_alphas)

    # Regular expression to pick attributes names.
    # Since in our case attributes names are the genomic positions (i.e. an integer number), we use the regex below
    import re
    pattern = "^\d+"

    # Auxiliary dictionary to create one pd.DataFrame for each sample, summing the shap values for each attribute.
    # Later, these dataframes will be appended together, resulting in the final df.
    dic={}

    for index, alpha in normalized_alphas.iteritems():
        # print(index)
        attr = re.match(pattern, index).group()
        if attr not in dic.keys():
            dic[attr] = (0.5 * alpha)
        else:
            dic[attr] += (0.5 * alpha)

    shap_original = pd.Series(dic)

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
"""
#######################################################################
Plot importances on genome
#######################################################################
"""

def plot_importances_genome(xgb_shap_values_df, rf_shap_values_df, alphas, analysis):

    xgb_shap_values_df = np.abs(xgb_shap_values_df)
    xgb_shap_values_df = np.sum(xgb_shap_values_df, axis=0)

    rf_shap_values_df = np.abs(rf_shap_values_df)
    rf_shap_values_df = np.sum(rf_shap_values_df, axis=0)

    fig, axes = plt.subplots(figsize=(10, 8), nrows=3, ncols=1, sharex=True)

    axes[0].scatter(xgb_shap_values_df.index, xgb_shap_values_df, c='blue')
    axes[0].set_title("")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("XGBoost importances")
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[0].set_yticklabels([])

    axes[1].scatter(rf_shap_values_df.index, rf_shap_values_df, c='blue')
    axes[1].set_title("")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Random Forest importances")
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[1].set_yticklabels([])

    axes[2].scatter(alphas.index, alphas, c='blue')
    axes[2].set_title("")
    axes[2].set_xlabel("Feature position (nucleotide position)")
    axes[2].set_ylabel("Logistic Regression importances")
    axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[2].set_yticklabels([])


    fig.tight_layout();

    fig.savefig(fig_dir+'/{}_overview_importances.png'.format(analysis), format='png', dpi=300, transparent=False)

"""
#######################################################################
Summary Plots
#######################################################################
"""
def importance_summary(xgb_shap_values_df, rf_shap_values_df, alphas, analysis):
    xgb_summary = get_sorted_importances(xgb_shap_values_df)
    rf_summary = get_sorted_importances(rf_shap_values_df)
    sorted_alphas = alphas.sort_values(ascending=False)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 8))

    ax[0].barh(xgb_summary.index[0:10], xgb_summary[0:10])
    ax[0].set_yticks(xgb_summary.index[0:10])
    ax[0].set_yticklabels(xgb_summary.index[0:10])
    ax[0].invert_yaxis()  # labels read top-to-bottom
    ax[0].set_xlabel('Feature Importance')
    ax[0].set_title('XGBoosting')
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax[1].barh(rf_summary.index[0:10], rf_summary[0:10])
    ax[1].set_yticks(rf_summary.index[0:10])
    ax[1].set_yticklabels(rf_summary.index[0:10])
    ax[1].invert_yaxis()  # labels read top-to-bottom
    ax[1].set_xlabel('Feature Importance')
    ax[1].set_title('Random Forest')
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax[2].barh(sorted_alphas.index[0:10], sorted_alphas[0:10])
    ax[2].set_yticks(sorted_alphas.index[0:10])
    ax[2].set_yticklabels(sorted_alphas.index[0:10])
    ax[2].invert_yaxis()  # labels read top-to-bottom
    ax[2].set_xlabel('Feature Importance')
    ax[2].set_title('Logistic Regression')
    ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    fig.tight_layout();

    fig.savefig(fig_dir+"/{}_summary.png".format(analysis), format='png', dpi=300, transparent=False)

    return (xgb_summary, rf_summary, sorted_alphas)

"""
#######################################################################
Merge all results in one series

Gets the top 10 most important nucleotide positions from each one of the 3 ML algorithms.
Normalizes their importance values (in each array, divides each value by the total sum of that array).
Merges into one array.
Normalizes again.
#######################################################################
"""
def get_merged_results(xgb_summary, rf_summary, sorted_alphas, analysis, min_importance_percentage=0.1):
    # xgb_summary_top = xgb_summary[:top]
    # rf_summary_top = rf_summary[:top]
    # sorted_alphas_top = sorted_alphas[:top]
    xgb_summary_top = xgb_summary
    rf_summary_top = rf_summary
    sorted_alphas_top = sorted_alphas

    xgb_sum = xgb_summary_top.sum()
    rf_sum = rf_summary_top.sum()
    sorted_alphas_sum = sorted_alphas_top.sum()

    xgb_summary_top = xgb_summary_top/xgb_sum
    rf_summary_top = rf_summary_top/rf_sum
    sorted_alphas_top = sorted_alphas_top/sorted_alphas_sum

    results_top = pd.concat([xgb_summary_top, rf_summary_top, sorted_alphas_top], axis=0)
    results_dic = {}

    for pos, imp in results_top.iteritems():
        if pos not in results_dic:
            results_dic[pos] = imp
        else:
            results_dic[pos] += imp

    results_all = pd.Series(results_dic)
    r_sum = results_all.sum()
    results_all = results_all/r_sum
    results_all = results_all.sort_values(ascending=False)


    # Keeps only those that have an importance up to 10% of the first attribute's importance.
    min_imp = results_all[0]*min_importance_percentage
    last_imp_pos = 0
    previous_pos = 0
    for pos, imp in results_all.iteritems():
        if imp <= min_imp:
            last_imp_pos = previous_pos
            break
        previous_pos += 1

    results_all = results_all[:previous_pos]


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), constrained_layout=True)

    ax.barh(results_all.index, results_all)
    ax.set_yticks(results_all.index)
    ax.set_yticklabels(results_all.index)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Feature Importance')
    # ax.set_title('HIV {} - {} most important genomic positions found by XGBoost, Random Forest and Modified Logistic Regression together.'.format(subtype, top))
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    fig.suptitle('Human Yellow Fever Infection Severity Analysis - {} most important genomic positions found by XGBoost, Random Forest and Modified Logistic Regression together.'.format(results_all.shape[0]), fontsize=16)

    # fig.tight_layout();

    fig.savefig(fig_dir+"/{}_combined_{}_summary.png".format(analysis, results_all.shape[0]), format='png', dpi=300, transparent=False)

    return results_all

"""
#######################################################################
Compare SNVs found by machine learning
Compare class 1 (high Ct) with class 0 (low Ct) and Alouatta samples.
Returns a table with the comparisons for the most important nucleotide positions found by the 3 ML algorithms
#######################################################################
"""
def validate_SNV(seq_df, imp_merged, size=50):

    # Serious human samples
    h1 = seq_df[seq_df["Class"]==1]
    # Columns corresponding to top feature importances
    h1 = h1.loc[:, np.array(list(imp_merged.index)).astype("int")]

    # Non-serious human samples
    h0 = seq_df[seq_df["Class"]==0]
    # Columns corresponding to top feature importances
    h0 = h0.loc[:, np.array(list(imp_merged.index)).astype("int")]

    # Arrays to keep the hallmark nucleotide values
    # The most frequent nucleotide for class 0
    # The differing nucleotide for class 1
    seq_c1 = h0.iloc[0,:].copy()
    seq_c0 = h0.iloc[0,:].copy()

    seq_c1[:]='NULL'
    seq_c0[:]='NULL'

    seq_c1.name='Serious'
    seq_c0.name='Non-serious'

    # For each nucleotide, gets the most frequent in class 0 and in Alouatta.
    # For class 1, gets the nucleotide that is different from the most frequent in class 0.
    for col, value in seq_c1.iteritems():
        nn_count_0 = h0.loc[:,col].value_counts()
        index = pd.Index(nn_count_0).get_loc(nn_count_0.max())
        nn_0 = nn_count_0.index[index]
        seq_c0[col] = nn_0

        nn_count_1 = h1.loc[:,col].value_counts()
        for nn, count in nn_count_1.iteritems():
            if nn != nn_0:
                seq_c1[col] = nn
        if seq_c1[col] == 'NULL':
            seq_c1[col] = nn_0

    # Creates a dataframe comparing the results
    df1 = pd.DataFrame(seq_c0).T
    df2 = pd.DataFrame(seq_c1).T

    table = pd.concat([df1, df2], axis=0)

    table = table.iloc[:, :size]

    return (table)

"""
#######################################################################
Plot confusion matrix
#######################################################################
"""
# print(sklearn.__version__)
# cm = confusion_matrix(y_test, y_test_pred)


def plot_confusion_matrix(y_true, y_pred, classes, method,
                          dataset,
                          analysis,
                          normalize=True,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title='{0} Confusion Matrix - {1}'.format(method, dataset)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / cm.sum()
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True Class',
           xlabel='Predicted Class')
    ax.tick_params(grid_alpha=0)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(fig_dir+'/{}_confusion_{}_{}.png'.format(analysis, method, dataset), format='png', dpi=300, transparent=False)
    return ax












"""
#######################################################################
MAIN
#######################################################################
""""""
#######################################################################
MAIN
#######################################################################
""""""
#######################################################################
MAIN
#######################################################################
""""""
#######################################################################
MAIN
#######################################################################
""""""
#######################################################################
MAIN
#######################################################################
""""""
#######################################################################
MAIN
#######################################################################
"""
"""
#######################################################################
MAIN
#######################################################################
"""

""" /////////////////////////////////////////////////////////////////////// """

""" SET THE STAGE... """

# Create OUTPUT dir inside DATA dir, where all processed data, figures, tbles, ect will be stored

working_dir = '/Users/alvarosalgado/Google Drive/Bioinformática/!Qualificação_alvaro/YFV'

if os.path.isdir(working_dir+'/2_OUTPUT')==False:
    os.mkdir(working_dir+'/2_OUTPUT/')
if os.path.isdir(working_dir+'/2_OUTPUT/FIGURES/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/FIGURES/')
if os.path.isdir(working_dir+'/2_OUTPUT/TABLES/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/TABLES/')
if os.path.isdir(working_dir+'/2_OUTPUT/PICKLE/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/PICKLE/')

out_dir = working_dir+'/2_OUTPUT'
fig_dir = working_dir+'/2_OUTPUT/FIGURES'
tab_dir = working_dir+'/2_OUTPUT/TABLES'
pik_dir = working_dir+'/2_OUTPUT/PICKLE'
data_dir = working_dir+'/1_DATA/Human_Analisys'

log_file = out_dir+'/LOG_YFV_HUMAN_MAIN_{0}.txt'.format(datetime.datetime.now())
with open(log_file, 'w') as log:
    x = datetime.datetime.now()
    log.write('LOG file for HUMAN YFV MAIN\n{0}\n\n'.format(x))



analysis = 'HUMAN'
# Data inmport
# %%
pickle_ohe = pik_dir+'/human_YFV_seq_ohe_df.pkl'
pickle_seqdf = pik_dir+ '/human_YFV_seq_df.pkl'
pickle_seqdforiginal = pik_dir+ '/human_YFV_original_seq_df.pkl'

(seq_df, ohe_df, seqdforiginal) = get_data(pickle_seqdf, pickle_ohe, pickle_seqdforiginal)


# Select which part of the dataset I'll use
# %%
ohe_df_yibra = ohe_df.loc[ohe_df["Dataset"] == "Yibra", :]
ohe_df_yibra["Class"].sum()

ohe_df_mari = ohe_df.loc[ohe_df["Dataset"] == "Marielton", :]
ohe_df_mari["Class"].sum()

ohe_df_tcura = ohe_df.loc[ohe_df["Dataset"] == "T_cura", :]
ohe_df_tcura["Class"].sum()

ohe_df_tob = ohe_df.loc[ohe_df["Dataset"] == "T_obitos", :]
ohe_df_tob["Class"].sum()

dataframes = [ohe_df_yibra, ohe_df_mari]
ohe_df_use = pd.concat(dataframes)

datasets_used = ohe_df_use["Dataset"].unique()

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()

    log.write("{0}\nDatasets for training:\n".format(x))
    for dataset in datasets_used:
        log.write("{0}, ".format(dataset))
    log.write("\n\n")

# Prepare data for training and testing
# %%

test_size = 0.5
(X, y, X_train, X_test, y_train, y_test, scale_pos_weight) = get_train_test_split(ohe_df_use, test_size=test_size)

positive = y_train.sum()
negative = len(y_train) - positive
scale_pos_weight = negative/positive


with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nTest Dataset Size: {1}%\n\n".format(x, test_size*100))

y.shape
y.sum()

y_train.shape
y_train.sum()

# DataFrame to keep scores
# %%
index_names = [['XGB', 'RF', 'MLR'], ['Test Dataset', 'Full Dataset']]
multi_index = pd.MultiIndex.from_product(index_names, names=['Method', 'Dataset'])
performance_df = pd.DataFrame(columns=['ROC-AUC', 'Accuracy', 'Precision'], index=multi_index)


performance_df.to_csv(tab_dir+'/{}_PERFORMANCE_models.csv'.format(analysis), index=True)

# XGBoost Grid Search
# %%

# Parameter grid for XGBoost
params = {
        'subsample': [1, 0.8],
        'colsample_bytree': [0.3, 1],
        'max_depth': [1, 3, 10, 100],
        'learning_rate': [0.001, 0.1, 1],
        'n_estimators': [100, 10000]
        }

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nParameters used for XGBoost grid search CV:\n{1}\n\n".format(x, params))
positive_weight = 0.01
grid = grid_cv_xgb(X_train, y_train, scale_pos_weight, params, analysis, folds = 5)
best_params = grid.best_params_

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nBest Parameters:\n{1}\n\n".format(x, best_params))


results = pd.DataFrame(grid.cv_results_)
results.to_csv(tab_dir+'/{0}_xgb-grid-search-results-01_{1}.csv'.format(analysis, datetime.datetime.now()), index=False)
results["mean_test_roc_auc"].unique()

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nThe grid search CV found in XGBoost that resulted in the following ROC-AUC scores:\n{1}\n\n".format(x, results["mean_test_roc_auc"].unique()))
    log.write("Therefore, the best parameters chosen are:\n{0}\n\n".format(best_params))


# params_series = results.loc[results['mean_test_roc_auc'] == np.max(results['mean_test_roc_auc']), 'params']
# for p in params_series:
#     print(p)
#
# print(best_params)
# print(params_series[0])

# Train models
# %%
"""XGB------------------------------------------------------------------------"""

method = 'XGB'
with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nStarting {1} Model////////////////////////////////////\n\n".format(x, method))

best_params = {'colsample_bytree': 0.3,
 'learning_rate': 0.001,
 'max_depth': 1,
 'n_estimators': 100,
 'subsample': 1}

xgb = final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, best_params, analysis)

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nXGBoost Model:\n{1}\n\n".format(x, xgb))

# Probability predicted by model for
# test dataset and full dataset (train + test)
y_test_prob = np.array(xgb.predict_proba(X_test))
y_all_prob = np.array(xgb.predict_proba(X))

# Classification predicted by model for
# test dataset and full dataset (train + test)
y_test_pred = xgb.predict(X_test)
y_all_pred = xgb.predict(X)

from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_curve


print(classification_report(y_test,y_test_pred))

with open(out_dir+'/LOG_MAIN_ML.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\n'---Classification Report---'\n{1}\n\n".format(x, method))
    log.write("{0}\n\n".format(classification_report(y_test,y_test_pred)))


fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_prob[:,1])
fpr_all, tpr_all, thresholds_all = roc_curve(y, y_all_prob[:, 1])

score_roc_auc_test = roc_auc_score(y_test, y_test_prob[:,1])
score_roc_auc_all = roc_auc_score(y, y_all_prob[:, 1])

score_test = xgb.score(X_test, y_test)
score_all = xgb.score(X, y)

plot_roc(fpr_test, tpr_test, score_roc_auc_test, analysis, method, 'Test')

plot_roc(fpr_all, tpr_all, score_roc_auc_all, analysis, method, 'Full')

cm_test = confusion_matrix(y_test, y_test_pred)
cm_all = confusion_matrix(y, y_all_pred)

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nXGBoost Confusion Matrices:\n\nTest Dataset:\n{1}\n\nFull Dataset:\n{2}\n\n".format(x, cm_test, cm_all))

performance_df.loc[method, 'Test Dataset']['ROC-AUC'] = score_roc_auc_test
performance_df.loc[method, 'Full Dataset']['ROC-AUC'] = score_roc_auc_all
performance_df.loc[method, 'Test Dataset']['Accuracy'] = score_test
performance_df.loc[method, 'Full Dataset']['Accuracy'] = score_all
performance_df.loc[method, 'Test Dataset']['Precision'] = cm_test[1,1]/(cm_test[1,1] + cm_test[0,1])
performance_df.loc[method, 'Full Dataset']['Precision'] = cm_all[1,1]/(cm_all[1,1] + cm_all[0,1])


ax = plot_confusion_matrix(y_test, y_test_pred,
                            ['Non-Severe', 'Severe'],
                            method,
                            'test_dataset',
                            analysis,
                            normalize=False,
                            cmap=plt.cm.Blues)

ax = plot_confusion_matrix(y, y_all_pred,
                            ['Non-Severe', 'Severe'],
                            method,
                            'full_dataset',
                            analysis,
                            normalize=False,
                            cmap=plt.cm.Blues)





"""RF------------------------------------------------------------------------"""
#%%
method = 'RF'

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nStarting {1} Model////////////////////////////////////\n\n".format(x, method))

weight = {0: 1, 1: scale_pos_weight}

rf = RandomForestClassifier(n_estimators=100,
                            random_state=0,
                            max_features = 'auto',
                            bootstrap=True,
                            oob_score=True,
                            class_weight=weight)

rf.fit(X_train, y_train)

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nRandom Forest Model:\n{1}\n\n".format(x, rf))

y_test_prob = rf.predict_proba(X_test)
y_all_prob = rf.predict_proba(X)

y_test_pred = rf.predict(X_test)
y_all_pred = rf.predict(X)

s1 = pd.Series(y_test_pred, name='predictions')
s2 = pd.Series(y_test.values, name='truth')
preds = pd.concat([s1, s2], axis=1)
preds['error'] = abs(preds['truth']-preds['predictions'])
preds['error'].sum()
preds[preds['truth']==1]

fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_prob[:, 1])
fpr_all, tpr_all, thresholds_all = roc_curve(y, y_all_prob[:, 1])

score_roc_auc_test = roc_auc_score(y_test, y_test_pred)
score_roc_auc_all = roc_auc_score(y, y_all_pred)

score_test = rf.score(X_test, y_test)
score_all = rf.score(X, y)


print(classification_report(y_test,y_test_pred))

with open(out_dir+'/LOG_MAIN_ML.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\n'---Classification Report---'\n{1}\n\n".format(x, method))
    log.write("{0}\n\n".format(classification_report(y_test,y_test_pred)))


plot_roc(fpr_test, tpr_test, score_roc_auc_test, analysis, method, 'Test')

plot_roc(fpr_all, tpr_all, score_roc_auc_all, analysis, method, 'Full')


cm_test = confusion_matrix(y_test, y_test_pred)
cm_all = confusion_matrix(y, y_all_pred)

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nRandom Forest Confusion Matrices:\n\nTest Dataset:\n{1}\n\nFull Dataset:\n{2}\n\n".format(x, cm_test, cm_all))

performance_df.loc[method, 'Test Dataset']['ROC-AUC'] = score_roc_auc_test
performance_df.loc[method, 'Full Dataset']['ROC-AUC'] = score_roc_auc_all
performance_df.loc[method, 'Test Dataset']['Accuracy'] = score_test
performance_df.loc[method, 'Full Dataset']['Accuracy'] = score_all
performance_df.loc[method, 'Test Dataset']['Precision'] = cm_test[1,1]/(cm_test[1,1] + cm_test[0,1])
performance_df.loc[method, 'Full Dataset']['Precision'] = cm_all[1,1]/(cm_all[1,1] + cm_all[0,1])

ax = plot_confusion_matrix(y_test, y_test_pred,
                            ['Non-Severe', 'Severe'],
                            method,
                            'test_dataset',
                            analysis,
                            normalize=False,
                            cmap=plt.cm.Greens)

ax = plot_confusion_matrix(y, y_all_pred,
                            ['Non-Severe', 'Severe'],
                            method,
                            'full_dataset',
                            analysis,
                            normalize=False,
                            cmap=plt.cm.Greens)

"""MLR------------------------------------------------------------------------"""
#%%
# Apply modified (regularized) Logistic Regression
# The module (script I created) is derived from Francielly Rodrigues Ph.D. work.

import Logistic_regression_modified as lr
method = 'MLR'

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nStarting {1} Model////////////////////////////////////\n\n".format(x, method))

(alphas, ohe_normalized_alphas, e, ave_MSE) = logistic_regression(X_train, y_train, X, k=10)

#%%
# The 'ohe_normalized_alphas' from 'logistic_regression' doesn't have
# aplha_0, which is needed to perform LR_predict.
# This is why I have to create 'alphas_plot'.
# It must be created with X_train, so that the test on the
# 'test_dataset' is valid.

score = 1 - ave_MSE
alphas_plot = lr.logistic_regression(X_train, y_train)
alphas_plot.shape

y_test_prob, y_test_pred = lr.LR_predict(alphas_plot, X_test)

y_all_prob, y_all_pred = lr.LR_predict(alphas_plot, X)

#%%

score_test = 1 - sum(np.sqrt((y_test_pred - y_test)**2))/len(y_test)
score_all = 1 - sum(np.sqrt((y_all_pred - y)**2))/len(y)

fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_pred)
fpr_all, tpr_all, thresholds_all = roc_curve(y, y_all_pred)

score_roc_auc_test = roc_auc_score(y_test, y_test_pred)
score_roc_auc_all = roc_auc_score(y, y_all_pred)

print(classification_report(y_test,y_test_pred))

with open(out_dir+'/LOG_MAIN_ML.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\n'---Classification Report---'\n{1}\n\n".format(x, method))
    log.write("{0}\n\n".format(classification_report(y_test,y_test_pred)))

plot_roc(fpr_test, tpr_test, score_roc_auc_test, analysis, method, 'Test')

plot_roc(fpr_all, tpr_all, score_roc_auc_all, analysis, method, 'Full')

cm_test = confusion_matrix(y_test, y_test_pred)
cm_all = confusion_matrix(y, y_all_pred)

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nMod Logistic Regression Confusion Matrices:\n\nTest Dataset:\n{1}\n\nFull Dataset:\n{2}\n\n".format(x, cm_test, cm_all))

performance_df.loc[method, 'Test Dataset']['ROC-AUC'] = score_roc_auc_test
performance_df.loc[method, 'Full Dataset']['ROC-AUC'] = score_roc_auc_all
performance_df.loc[method, 'Test Dataset']['Accuracy'] = score_test
performance_df.loc[method, 'Full Dataset']['Accuracy'] = score_all
performance_df.loc[method, 'Test Dataset']['Precision'] = cm_test[1,1]/(cm_test[1,1] + cm_test[0,1])
performance_df.loc[method, 'Full Dataset']['Precision'] = cm_all[1,1]/(cm_all[1,1] + cm_all[0,1])

ax = plot_confusion_matrix(y_test, y_test_pred,
                            ['Non-Severe', 'Severe'],
                            method,
                            'test_dataset',
                            analysis,
                            normalize=False,
                            cmap=plt.cm.Greys)

ax = plot_confusion_matrix(y, y_all_pred,
                            ['Non-Severe', 'Severe'],
                            method,
                            'full_dataset',
                            analysis,
                            normalize=False,
                            cmap=plt.cm.Greys)




with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nPerformance Table\n\n{1}\n\n".format(x, performance_df))

"""
#######################################################################
SHAP
#######################################################################
"""
# Use SHAP to explain models
# %%
(xgb_explainer, rf_explainer, xgb_shap_values, rf_shap_values) = get_explainer(xgb, rf, X_train)

# xgb_shap_values.shape
# rf_shap_values.shape
# X_train.columns
# X_train.index
rf_shap_values_df = pd.DataFrame(rf_shap_values,
                                index=X_train.index,
                                columns=X_train.columns)

xgb_shap_values_df = pd.DataFrame(xgb_shap_values,
                                 index=X_train.index,
                                 columns=X_train.columns)

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nExample SHAP values for Random Forest in One Hot Encoded format:\n\n{1}\n\n".format(x, rf_shap_values_df.iloc[0:7, 0:2]
))


rf_shap_values_df = ohe_inverse(rf_shap_values_df)
xgb_shap_values_df = ohe_inverse(xgb_shap_values_df)

with open(out_dir+'/log_MAIN_YFV_HUMAN.txt', 'a') as log:
    x = datetime.datetime.now()
    log.write("{0}\nExample SHAP values for Random Forest in Original Genomic Position format:\n\n{1}\n\n".format(x, rf_shap_values_df.iloc[0:7, 0:1]
))
"""
#######################################################################
Plot results
#######################################################################
"""
# Plot resulting feature importances
# %%
plot_importances_genome(xgb_shap_values_df, rf_shap_values_df, alphas, analysis)

# Get importances values and genomic locations
(xgb_summary, rf_summary, sorted_alphas) = importance_summary(xgb_shap_values_df, rf_shap_values_df, alphas, analysis)

"""
#######################################################################
Analyze results
#######################################################################
"""

imp_merged = get_merged_results(xgb_summary, rf_summary, sorted_alphas, analysis, 0.01)


"""//////////////////////////////////////////////"""

#%%
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord
import ref_genome_polyprot_toolbox as g_tool

case = "HUMAN_YFV"

ref_genome_file = working_dir+'/3_REFERENCE/YFV_REF_NC_002031.fasta'
ref_polyprot_file = working_dir+'/3_REFERENCE/EDITED_YFV_REF_NC_002031.gb'
# querry_seq_file = data_dir+'/FUNED/ALIGNED_FUNED_SARS-CoV-2.fasta'

ref_genome = SeqIO.read(ref_genome_file, "fasta").lower()
ref_polyprot = SeqIO.read(ref_polyprot_file, "genbank")
#%%


"""
Given a polyprotein genbank file (in '.gp' format), parses through
its features and returns a dictionary containing the proteins
names as keys and positions (start:end in "biopython location") as values.
"""
for feature in ref_polyprot.features:
    print(feature)
feature.type

#%%
dic_prot = {}
# These positions are ZERO based, even though the .gb file is ONE based.
for feature in ref_polyprot.features:
    if feature.type == 'mat_peptide':
        value = (feature.location)
        key = (feature.qualifiers['gene'][0])
        dic_prot[key] = value


dic_prot # here indexing starts at zero
#%%

proteins=[key for key in dic_prot.keys()]

prot_seqs_dic = {}

for prot in proteins:
    start = dic_prot[prot].start
    end = dic_prot[prot].end
    seq = ref_genome.seq[start:end]
    prot_seqs_dic[prot]=seq
#%%






# (1-1)//3
# (2-1)//3
# (3-1)//3
# (1-1)%3
# (2-1)%3
# (3-1)%3
#
# nnpos = 14408 # here indexing starts at one
#
# protein
# protein_start
# nnpos_on_prot = nnpos - protein_start
# aapos = ((nnpos_on_prot-1)//3)+1 # to start at position 1
# codon_pos = ((nnpos_on_prot-1)%3)+1 # 1, 2, 3
# codon_start = nnpos - codon_pos
# codon = Seq(ref_genome[codon_start:codon_start+3])
# aa = codon.translate()
#
#
# ref_genome[265:271] # here indexing starts at zero
# ref_genome[265:271].translate() # here indexing starts at zero




def prot_info(nnpos, dic_prot, ref_genome):
    utr = True
    for prot in dic_prot:
        if ((nnpos-1) >= dic_prot[prot].start and (nnpos-1) < dic_prot[prot].end):
            utr = False
            protein_start = int(dic_prot[prot].start)
            nnpos_on_prot = nnpos - protein_start # Here, the number you get is results in indexing the protein nucleotides starting at ONE. So, if nnposonprot=3, it is the third nucleotide in the protein, 1, 2, 3!
            aapos = ((nnpos_on_prot-1)//3)+1 # to start at position 1
            codon_pos = ((nnpos_on_prot-1)%3)+1 # 1, 2, 3
            codon_start = nnpos - codon_pos # Also considering index starting at ONE.
            codon = ref_genome[codon_start:codon_start+3]
            aa_ref = codon.translate()
            break
    if utr:
        prot = 'UTR'
        protein_start = 'UTR'
        nnpos_on_prot = 'UTR'
        aapos = 'UTR'
        codon_pos = 'UTR'
        codon_start = 'UTR'
        codon = SeqRecord('UTR')
        aa_ref = SeqRecord('UTR')
    return(prot, protein_start, nnpos_on_prot, aapos, codon, codon_pos, aa_ref)



# 14408-13467
#
#
#
#
#
# nnpos = 14408 # here indexing starts at one
# (prot, protein_start, nnpos_on_prot, aapos, codon, codon_pos, aa_ref) = prot_info(nnpos, dic_prot, ref_genome)
#
# str(aa_ref)
# str(aa_ref.seq)
# str(codon.seq)






cols = ['Rank', 'nn postition', 'Protein', 'nn position on protein', 'aa position on protein', 'aa reference', 'codon reference', 'aa variation', 'codon variation', 'SNV codon position (1, 2, 3)']
# dummy_sample = np.zeros(len(cols))

rank = 1
rows = []
for nnpos in imp_merged.index:
    nnpos = int(nnpos)
    (prot, ps, nnpos_on_prot, aapos, codon, codon_pos, aa_ref) = prot_info(nnpos, dic_prot, ref_genome)

    if prot is not 'UTR':
        codon_var = []
        aa_var = []

        nn_ref = ref_genome[nnpos-1]
        nn_diversity = seqdforiginal[nnpos].unique()

        for nn in nn_diversity:
            if nn is not nn_ref:
                if (nn is 'a') or (nn is 'c') or (nn is 't') or (nn is 'g'):
                    new_codon = list(str(codon.seq))
                    new_codon[codon_pos-1] = nn
                    new_codon = "".join((new_codon))
                    new_codon=Seq(new_codon)
                    codon_var.append(new_codon)

        for new_codon in codon_var:
            new_aa = new_codon.translate()
            aa_var.append(new_aa)

        l = [str(c) for c in codon_var]
        codon_var = l

        l = [str(aa) for aa in aa_var]
        aa_var = l
    else:
        aa_var = 'X'
        codon_var = 'xxx'




    row = [rank, nnpos, prot, nnpos_on_prot, aapos, str(aa_ref.seq), str(codon.seq), " ".join(aa_var), " ".join(codon_var), codon_pos]
    rows.append(row)
    rank +=1

SNV_RESULTS = pd.DataFrame(rows, index=imp_merged.index, columns=cols)

SNV_RESULTS.to_csv(out_dir+'/SNV_HUMAN_YFV_RESULTS.csv')
































# The analysis below and the results shown in "table" demonstrate the power of XGBoost. It only picked 3 features, and there was a total of 5 that really had any informative value. All the rest, that both RF and LR gave some importance (albeit small), have no information at all, given that they do not contain a nucleotide that is different from the most frequent one in the other class and in the Alouatta samples.
