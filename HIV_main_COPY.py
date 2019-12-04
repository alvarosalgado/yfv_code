#!/usr/bin/env python3
"""
From 'YFV_Ct_Callithrix_main_rev1.ipynb'

Edited to work with HIV data
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
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.utils.multiclass import unique_labels

import pickle

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

    ohe_df = ohe_df.sort_values(by='Treatment')

    return (seq_df, ohe_df)

"""
#######################################################################
Train and test splits

Separate data into train and test sets.
Since the dataset is small and imbalanced, I will separate only 10% for testing.
#######################################################################
"""
def get_train_test_split(ohe_df, test_size=0.1):
    # Get only the ohe nucleotide info in X
    X = ohe_df.drop(["Header","Subtype","Treatment"], axis=1)
    # The target class is Ct_Group (high or low)
    y = ohe_df["Treatment"]

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
                        return_train_score=True,
                        n_jobs=4,
                        cv=skf.split(X_train,y_train),
                        verbose=1,
                        refit='roc_auc')


    grid.fit(X_train, y_train)

    # best_params = grid.best_params_
    # results = pd.DataFrame(grid.cv_results_)
    # results.to_csv('./OUTPUT/xgb-grid-search-results-01.csv', index=False)

    return (grid)
"""
#######################################################################
Final XGBoost model
Fitting the final XGBoost with parameters found on grid_cv.
Use all training data.
Test on test data.
#######################################################################
"""
def final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, params, subtype):

    xgb = XGBClassifier(**params)
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

    fig1.savefig('../FIGURES/{}_final_xgb_model.png'.format(subtype), format='png', dpi=300, transparent=False)

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
def random_forest(X_train, y_train, X_test, y_test, cw, n=100):

    rf = RandomForestClassifier(n_estimators=n,
                                random_state=0,
                                max_features = 'auto',
                                bootstrap=True,
                                oob_score=True,
                                class_weight={0:1, 1:cw})
    rf.fit(X_train, y_train)

    y_pred_prob = rf.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

    roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
    score = rf.score(X_test, y_test)

    return (rf, roc_auc, rf.oob_score_, score, fpr, tpr)


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
def plot_roc(fpr, tpr, roc_auc, subtype, method):
    fig, axes = plt.subplots(figsize=(10, 8), nrows=1, ncols=1)
    lw = 2
    axes.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area ={0:.2f})'.format(roc_auc))
    axes.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')
    axes.set_title('Receiver operating characteristic')
    axes.legend(loc="lower right")
    fig.show()
    fig.savefig("../FIGURES/ROC_{0}_{1}.png".format(method, subtype), format='png', dpi=300, transparent=False)

"""
#######################################################################
Feature Importances
#######################################################################
"""
# load JS visualization code to notebook
shap.initjs()

def get_explainer(xgb, rf, X):
    # Creates the explainer based on the model.
    rf_explainer = shap.TreeExplainer(rf, data=X)
    rf_shap_values = rf_explainer.shap_values(X)
    rf_shap_values = rf_shap_values[0] # For the random forest model, the shap TreeExplainer returns 2 sets of values, one for class 1 and one for class 0. They are symmetric, so you can use either.

    # Creates the explainer based on the model.
    xgb_explainer = shap.TreeExplainer(xgb, data=X)
    xgb_shap_values = xgb_explainer.shap_values(X)



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
    # the pattern used in the dataset is X123_
    pattern = "^X\d+"

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
    pattern = "^X\d+"

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

def plot_importances_genome(xgb_shap_values_df, rf_shap_values_df, alphas, subtype):

    xgb_shap_values_df = np.abs(xgb_shap_values_df)
    xgb_shap_values_df = np.sum(xgb_shap_values_df, axis=0)

    rf_shap_values_df = np.abs(rf_shap_values_df)
    rf_shap_values_df = np.sum(rf_shap_values_df, axis=0)

    fig, axes = plt.subplots(figsize=(10, 8), nrows=3, ncols=1, sharex=True, constrained_layout=True)

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
    axes[2].set_xlabel("Feature position (aminoacid position)")
    axes[2].set_ylabel("Logistic Regression importances")
    axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[2].set_yticklabels([])

    fig.suptitle('HIV {}'.format(subtype), fontsize=16)

    # fig.tight_layout();

    fig.savefig('../FIGURES/{}_overview.png'.format(subtype), format='png', dpi=300, transparent=False)

"""
#######################################################################
Summary Plots
#######################################################################
"""
def importance_summary(xgb_shap_values_df, rf_shap_values_df, alphas, subtype):
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

    fig.suptitle('HIV {}'.format(subtype), fontsize=16)

    # fig.tight_layout();

    fig.savefig('../FIGURES/{}_summary.png'.format(subtype), format='png', dpi=300, transparent=False)

    return (xgb_summary, rf_summary, sorted_alphas)

"""
#######################################################################
Merge all results in one series

Gets the top 10 most important nucleotide positions from each one of the 3 ML algorithms.
Normalizes their importance values (in each array, divides each value by the total sum of that array).
Merges into one array.
Normalizes again.

!!!!!
Edited from original YFV script to consider only XGB and MLR.
Jose Lourenco is already investigating RF
!!!!!

#######################################################################
"""
def get_merged_results(xgb_summary, rf_summary, sorted_alphas, top, subtype):
    xgb_summary_top = xgb_summary[:top]
    rf_summary_top = rf_summary[:top]
    sorted_alphas_top = sorted_alphas[:top]

    xgb_sum = xgb_summary_top.sum()
    rf_sum = rf_summary_top.sum()
    sorted_alphas_sum = sorted_alphas_top.sum()

    # Normalization. Weighted averages
    xgb_summary_top = xgb_summary_top/xgb_sum
    rf_summary_top = rf_summary_top/rf_sum
    sorted_alphas_top = sorted_alphas_top/sorted_alphas_sum

    '''!!!!!
    Edited from original YFV script to consider only XGB and MLR.
    Jose Lourenco is already investigating RF
    !!!!!'''
    # results_top = pd.concat([xgb_summary_top, rf_summary_top, sorted_alphas_top], axis=0)
    # results_dic = {}

    results_top = pd.concat([xgb_summary_top, sorted_alphas_top], axis=0)
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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), constrained_layout=True)

    ax.barh(results_all.index[0:top], results_all[0:top])
    ax.set_yticks(results_all.index[0:top])
    ax.set_yticklabels(results_all.index[0:top])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Feature Importance')
    # ax.set_title('HIV {} - {} most important genomic positions found by XGBoost, Random Forest and Modified Logistic Regression together.'.format(subtype, top))
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    fig.suptitle('HIV {} - {} most important genomic positions found by XGBoost, Random Forest and Modified Logistic Regression together.'.format(subtype, top), fontsize=16)

    # fig.tight_layout();

    fig.savefig("../FIGURES/{}_combined_{}_summary.png".format(subtype, top), format='png', dpi=300, transparent=False)

    return results_all


"""
#######################################################################
Save SHAP values to CSV file
#######################################################################
"""
def shap_values_CSV(shap_values_df, method, subtype):
    values = np.abs(shap_values_df)
    values = np.sum(values, axis=0)
    values.to_csv('../OUTPUT/shap_values_{}_{}.csv'.format(method, subtype), index=True, header=['Score'])
"""
#######################################################################
Plot confusion matrix
#######################################################################
"""
def plot_confusion_matrix(y_true, y_pred, classes, subtype, method,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title='{} - {} Confusion Matrix - Test Set'.format(subtype, method)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

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
    fig.savefig('../FIGURES/{}_{}_confusion.png'.format(subtype, method), format='png', dpi=300, transparent=False)
    return ax

# y_true = [1, 0, 1, 0, 0, 1]
# y_pred = [1, 0, 1, 1, 1, 0]
# confusion_matrix(y_true, y_pred)
"""
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
"""























"""
#######################################################################
MAIN 02_AG
#######################################################################
"""

# Data inmport
# %%
subtype = '02_AG'
pickle_ohe = '../!DATA/{}_ohe.pkl'.format(subtype)
pickle_seqdf = '../!DATA/{}.pkl'.format(subtype)

(seq_df, ohe_df) = get_data(pickle_seqdf, pickle_ohe)
seq_df.shape

ohe_df['Treatment'] = ohe_df['Treatment'].astype(int)

# Prepare data for training and testing
(X, y, X_train, X_test, y_train, y_test, scale_pos_weight) = get_train_test_split(ohe_df, test_size=0.1)



ohe_df_explain = ohe_df.groupby('Treatment', group_keys=False).apply(lambda x: x.sample(1000))
ohe_df_explain.shape
# Get only the ohe nucleotide info in X
X_explain = ohe_df_explain.drop(["Header","Subtype","Treatment"], axis=1)
# The target class is Ct_Group (high or low)
y_explain = ohe_df_explain["Treatment"]

# DataFrame to keep scores
index_names = [['02_AG', 'A', 'B', 'C'], ['XGB', 'RF', 'MLR']]
multi_index = pd.MultiIndex.from_product(index_names, names=['Subtype', 'Method'])
performance_df = pd.DataFrame(columns=['ROC-AUC', 'Accuracy', 'Precision'], index=multi_index)
performance_df.loc['A', 'XGB']['Accuracy']


performance_df.to_csv('../OUTPUT/PERFORMANCE_hiv_ML_models.csv', index=True)


# %%
"""XGB-----------------------------------------------------------------------"""


'''
# GRID SEARCH CV
################################################################################
params = {
        'subsample': [0.8],
        'colsample_bytree': [0.3],
        'max_depth': [3, 50],
        'learning_rate': [0.0001, 0.1],
        'n_estimators': [500]
        }

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)

xgb = XGBClassifier(objective='binary:logistic', njobs=4, random_state=0, scale_pos_weight=scale_pos_weight)

grid = GridSearchCV(estimator=xgb,
                    param_grid=params,
                    scoring='roc_auc',
                    n_jobs=4,
                    cv=skf.split(X_train,y_train),
                    verbose=1)

grid.fit(X_train, y_train)

best_params = grid.best_params_
grid_results = pd.DataFrame(grid.cv_results_)
xgb = grid.best_estimator_

xgb = final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, best_params, subtype)

# save model to file
pickle.dump(xgb, open("../OUTPUT/xgb_{}.pickle.dat".format(subtype), "wb"))
################################################################################
'''

method = 'XGB'

# load model from file
xgb = pickle.load(open("../OUTPUT/xgb_{}.pickle.dat".format(subtype), "rb"))

params = xgb.get_xgb_params()

# plot XGBoost training curves
xgb = final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, params, subtype)

# Prediction probabilities for ROC curve plotting
y_pred_prob = xgb.predict_proba(X_test)

# False positive rate, True positive rate, Decision threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

# ROC-AUC and Accuracy score
roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
score = xgb.score(X_test, y_test)

# Plot of a ROC curve for a specific class
plot_roc(fpr, tpr, roc_auc, subtype, method)

y_pred = xgb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

performance_df.loc[subtype, method]['ROC-AUC'] = roc_auc
performance_df.loc[subtype, method]['Accuracy'] = score
performance_df.loc[subtype, method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])


ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Blues)



"""RF------------------------------------------------------------------------"""
# Random forest

method='RF'

# (rf, roc_auc_rf, rf_oob_score, score_rf, fpr_rf, tpr_rf) = random_forest(X_train, y_train, X_test, y_test, scale_pos_weight, n=500)
#
# # save model to file
# pickle.dump(rf, open("../OUTPUT/rf_{}.pickle.dat".format(subtype), "wb"))

# load model from file
rf = pickle.load(open("../OUTPUT/rf_{}.pickle.dat".format(subtype), "rb"))

y_pred_prob = rf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
score = rf.score(X_test, y_test)

plot_roc(fpr, tpr, roc_auc, subtype, method)

y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

performance_df.loc[subtype, method]['ROC-AUC'] = roc_auc
performance_df.loc[subtype, method]['Accuracy'] = score
performance_df.loc[subtype, method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])

ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Greens)
"""MLR-----------------------------------------------------------------------"""
# Logistic Regression
import Logistic_regression_modified as lr
method = 'MLR'
(alphas, ohe_normalized_alphas, e, ave_MSE) = logistic_regression(X_train, y_train, X, k=10)

score = 1 - ave_MSE

# alphas used for plotting ROC curve
alphas_plot = lr.logistic_regression(X_train, y_train)
alphas_plot.shape

y_pred_prob, y_pred = lr.LR_predict(alphas_plot, X_test)
X_test.drop('alpha_0', axis=1, inplace=True)

# y_pred_prob_xgb = xgb.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot of a ROC curve for a specific class
plot_roc(fpr, tpr, roc_auc, subtype, method='MLR')


cm = confusion_matrix(y_test, y_pred)

performance_df.loc[subtype, method]['ROC-AUC'] = roc_auc
performance_df.loc[subtype, method]['Accuracy'] = score
performance_df.loc[subtype, method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])

ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Greys)
"""SHAP----------------------------------------------------------------------"""
# Use SHAP to explain models
# %%
X_explain.shape
rf_explainer = shap.TreeExplainer(rf, data=None)
rf_shap_values = rf_explainer.shap_values(X_explain, approximate=True)
rf_shap_values = rf_shap_values[0] # For the random forest model, the shap TreeExplainer returns 2 sets of values, one for class 1 and one for class 0. They are symmetric, so you can use either.

# Creates the explainer based on the model.
xgb_explainer = shap.TreeExplainer(xgb, data=None)
# xgb_explainer = shap.TreeExplainer(xgb, data=X_explain, feature_dependence="independent")
xgb_shap_values = xgb_explainer.shap_values(X_explain, approximate=True)

with open('../OUTPUT/xgb_explainer_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(xgb_explainer, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/rf_explainer_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(rf_explainer, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/xgb_shap_values_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(xgb_shap_values, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/rf_shap_values_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(rf_shap_values, f, pickle.HIGHEST_PROTOCOL)

xgb_shap_values = pickle.load(open("../OUTPUT/xgb_shap_values_{}.pickle.dat".format(subtype), "rb"))

rf_shap_values = pickle.load(open("../OUTPUT/rf_shap_values_{}.pickle.dat".format(subtype), "rb"))


rf_shap_values_df = pd.DataFrame(rf_shap_values,
                                index=X_explain.index,
                                columns=X_explain.columns)

xgb_shap_values_df = pd.DataFrame(xgb_shap_values,
                                 index=X_explain.index,
                                 columns=X_explain.columns)

rf_shap_values_df = ohe_inverse(rf_shap_values_df)

xgb_shap_values_df = ohe_inverse(xgb_shap_values_df)

shap_values_CSV(xgb_shap_values_df, 'xgb', subtype)

shap_values_CSV(rf_shap_values_df, 'rf', subtype)

alphas.to_csv('../OUTPUT/shap_values_mlr_{}.csv'.format(subtype), index=True, header=['Score'])



"""RESULTS-------------------------------------------------------------------"""
# Plot resulting feature importances
# %%
plot_importances_genome(xgb_shap_values_df, rf_shap_values_df, alphas, subtype)

# Get importances values and genomic locations
(xgb_summary, rf_summary, sorted_alphas) = importance_summary(xgb_shap_values_df, rf_shap_values_df, alphas, subtype)


imp_merged = get_merged_results(xgb_summary, rf_summary, sorted_alphas, 30, subtype)

imp_merged.to_csv('../OUTPUT/{}_importances_merged_XGB_MLR.csv'.format(subtype), index=True, header=['Score'])




























"""
#######################################################################
MAIN subtype_A
#######################################################################
"""

# Data inmport
# %%
subtype = 'subtype_A'
pickle_ohe = '../!DATA/{}_ohe.pkl'.format(subtype)
pickle_seqdf = '../!DATA/{}.pkl'.format(subtype)

(seq_df, ohe_df) = get_data(pickle_seqdf, pickle_ohe)
seq_df.shape

ohe_df['Treatment'] = ohe_df['Treatment'].astype(int)

# Prepare data for training and testing
(X, y, X_train, X_test, y_train, y_test, scale_pos_weight) = get_train_test_split(ohe_df, test_size=0.1)



ohe_df_explain = ohe_df.groupby('Treatment', group_keys=False).apply(lambda x: x.sample(1000))
ohe_df_explain.shape
# Get only the ohe nucleotide info in X
X_explain = ohe_df_explain.drop(["Header","Subtype","Treatment"], axis=1)
# The target class is Ct_Group (high or low)
y_explain = ohe_df_explain["Treatment"]

# DataFrame to keep scores
# Run only in the beginning, then save at the end.
# index_names = [['02_AG', 'A', 'B', 'C'], ['XGB', 'RF', 'MLR']]
# multi_index = pd.MultiIndex.from_product(index_names, names=['Subtype', 'Method'])
# performance_df = pd.DataFrame(columns=['ROC-AUC', 'Accuracy', 'Precision'], index=multi_index)
# performance_df.loc['A', 'XGB']['Accuracy']

# %%
"""XGB-----------------------------------------------------------------------"""



# GRID SEARCH CV
################################################################################
params = {
        'subsample': [0.8],
        'colsample_bytree': [0.3],
        'max_depth': [3, 50],
        'learning_rate': [0.0001, 0.1],
        'n_estimators': [500]
        }

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)

xgb = XGBClassifier(objective='binary:logistic', njobs=4, random_state=0, scale_pos_weight=scale_pos_weight)

grid = GridSearchCV(estimator=xgb,
                    param_grid=params,
                    scoring='roc_auc',
                    n_jobs=4,
                    cv=skf.split(X_train,y_train),
                    verbose=1)

grid.fit(X_train, y_train)

best_params = grid.best_params_
grid_results = pd.DataFrame(grid.cv_results_)
xgb = grid.best_estimator_

xgb = final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, best_params, subtype)

# save model to file
pickle.dump(xgb, open("../OUTPUT/xgb_{}.pickle.dat".format(subtype), "wb"))
################################################################################


method = 'XGB'

# load model from file
xgb = pickle.load(open("../OUTPUT/xgb_{}.pickle.dat".format(subtype), "rb"))

params = xgb.get_xgb_params()

# plot XGBoost training curves
xgb = final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, params, subtype)

# Prediction probabilities for ROC curve plotting
y_pred_prob = xgb.predict_proba(X_test)

# False positive rate, True positive rate, Decision threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

# ROC-AUC and Accuracy score
roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
score = xgb.score(X_test, y_test)

# Plot of a ROC curve for a specific class
plot_roc(fpr, tpr, roc_auc, subtype, method)

y_pred = xgb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

performance_df.loc[ 'A', method]['ROC-AUC'] = roc_auc
performance_df.loc['A', method]['Accuracy'] = score
performance_df.loc['A', method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])


ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Blues)



"""RF------------------------------------------------------------------------"""
# Random forest

method='RF'

# (rf, roc_auc_rf, rf_oob_score, score_rf, fpr_rf, tpr_rf) = random_forest(X_train, y_train, X_test, y_test, scale_pos_weight, n=500)
#
# # save model to file
# pickle.dump(rf, open("../OUTPUT/rf_{}.pickle.dat".format(subtype), "wb"))

# load model from file
rf = pickle.load(open("../OUTPUT/rf_{}.pickle.dat".format(subtype), "rb"))

y_pred_prob = rf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
score = rf.score(X_test, y_test)

plot_roc(fpr, tpr, roc_auc, subtype, method)

y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

performance_df.loc['A', method]['ROC-AUC'] = roc_auc
performance_df.loc['A', method]['Accuracy'] = score
performance_df.loc['A', method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])

ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Greens)
"""MLR-----------------------------------------------------------------------"""
# Logistic Regression
import Logistic_regression_modified as lr
method = 'MLR'
(alphas, ohe_normalized_alphas, e, ave_MSE) = logistic_regression(X_train, y_train, X, k=10)

score = 1 - ave_MSE

# alphas used for plotting ROC curve
alphas_plot = lr.logistic_regression(X_train, y_train)
alphas_plot.shape

y_pred_prob, y_pred = lr.LR_predict(alphas_plot, X_test)
X_test.drop('alpha_0', axis=1, inplace=True)

# y_pred_prob_xgb = xgb.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot of a ROC curve for a specific class
plot_roc(fpr, tpr, roc_auc, subtype, method='MLR')


cm = confusion_matrix(y_test, y_pred)

performance_df.loc['A', method]['ROC-AUC'] = roc_auc
performance_df.loc['A', method]['Accuracy'] = score
performance_df.loc['A', method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])

ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Greys)
"""SHAP----------------------------------------------------------------------"""
# Use SHAP to explain models
# %%
X_explain.shape
rf_explainer = shap.TreeExplainer(rf, data=None)
rf_shap_values = rf_explainer.shap_values(X_explain, approximate=True)
rf_shap_values = rf_shap_values[0] # For the random forest model, the shap TreeExplainer returns 2 sets of values, one for class 1 and one for class 0. They are symmetric, so you can use either.

# Creates the explainer based on the model.
xgb_explainer = shap.TreeExplainer(xgb, data=None)
# xgb_explainer = shap.TreeExplainer(xgb, data=X_explain, feature_dependence="independent")
xgb_shap_values = xgb_explainer.shap_values(X_explain, approximate=True)

with open('../OUTPUT/xgb_explainer_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(xgb_explainer, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/rf_explainer_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(rf_explainer, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/xgb_shap_values_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(xgb_shap_values, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/rf_shap_values_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(rf_shap_values, f, pickle.HIGHEST_PROTOCOL)

xgb_shap_values = pickle.load(open("../OUTPUT/xgb_shap_values_{}.pickle.dat".format(subtype), "rb"))

rf_shap_values = pickle.load(open("../OUTPUT/rf_shap_values_{}.pickle.dat".format(subtype), "rb"))


rf_shap_values_df = pd.DataFrame(rf_shap_values,
                                index=X_explain.index,
                                columns=X_explain.columns)

xgb_shap_values_df = pd.DataFrame(xgb_shap_values,
                                 index=X_explain.index,
                                 columns=X_explain.columns)

rf_shap_values_df = ohe_inverse(rf_shap_values_df)

xgb_shap_values_df = ohe_inverse(xgb_shap_values_df)

shap_values_CSV(xgb_shap_values_df, 'xgb', subtype)

shap_values_CSV(rf_shap_values_df, 'rf', subtype)

alphas.to_csv('../OUTPUT/shap_values_mlr_{}.csv'.format(subtype), index=True, header=['Score'])



"""RESULTS-------------------------------------------------------------------"""
# Plot resulting feature importances
# %%
plot_importances_genome(xgb_shap_values_df, rf_shap_values_df, alphas, subtype)

# Get importances values and genomic locations
(xgb_summary, rf_summary, sorted_alphas) = importance_summary(xgb_shap_values_df, rf_shap_values_df, alphas, subtype)


imp_merged = get_merged_results(xgb_summary, rf_summary, sorted_alphas, 30, subtype)

imp_merged.to_csv('../OUTPUT/{}_importances_merged_XGB_MLR.csv'.format(subtype), index=True, header=['Score'])





















"""
#######################################################################
MAIN subtype_B
#######################################################################
"""

# Data inmport
# %%
subtype = 'subtype_B'
pickle_ohe = '../!DATA/{}_ohe.pkl'.format(subtype)
pickle_seqdf = '../!DATA/{}.pkl'.format(subtype)

(seq_df, ohe_df) = get_data(pickle_seqdf, pickle_ohe)
seq_df.shape

ohe_df['Treatment'] = ohe_df['Treatment'].astype(int)

# Prepare data for training and testing
(X, y, X_train, X_test, y_train, y_test, scale_pos_weight) = get_train_test_split(ohe_df, test_size=0.1)



ohe_df_explain = ohe_df.groupby('Treatment', group_keys=False).apply(lambda x: x.sample(1000))
ohe_df_explain.shape
# Get only the ohe nucleotide info in X
X_explain = ohe_df_explain.drop(["Header","Subtype","Treatment"], axis=1)
# The target class is Ct_Group (high or low)
y_explain = ohe_df_explain["Treatment"]

# DataFrame to keep scores
# Run only in the beginning, then save at the end.
# index_names = [['02_AG', 'A', 'B', 'C'], ['XGB', 'RF', 'MLR']]
# multi_index = pd.MultiIndex.from_product(index_names, names=['Subtype', 'Method'])
# performance_df = pd.DataFrame(columns=['ROC-AUC', 'Accuracy', 'Precision'], index=multi_index)
# performance_df.loc['A', 'XGB']['Accuracy']

# %%
"""XGB-----------------------------------------------------------------------"""



# GRID SEARCH CV
################################################################################
params = {
        'subsample': [0.8],
        'colsample_bytree': [0.3],
        'max_depth': [3, 25, 50],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100]
        }

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)

xgb = XGBClassifier(objective='binary:logistic', njobs=4, random_state=0, scale_pos_weight=scale_pos_weight)

grid = GridSearchCV(estimator=xgb,
                    param_grid=params,
                    scoring='roc_auc',
                    n_jobs=4,
                    cv=skf.split(X_train,y_train),
                    verbose=1)

grid.fit(X_train, y_train)

best_params = grid.best_params_
grid_results = pd.DataFrame(grid.cv_results_)
xgb = grid.best_estimator_

xgb = final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, best_params, subtype)

# save model to file
pickle.dump(xgb, open("../OUTPUT/xgb_{}.pickle.dat".format(subtype), "wb"))
################################################################################


method = 'XGB'

# load model from file
xgb = pickle.load(open("../OUTPUT/xgb_{}.pickle.dat".format(subtype), "rb"))

params = xgb.get_xgb_params()

# plot XGBoost training curves
xgb = final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, params, subtype)

# Prediction probabilities for ROC curve plotting
y_pred_prob = xgb.predict_proba(X_test)

# False positive rate, True positive rate, Decision threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

# ROC-AUC and Accuracy score
roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
score = xgb.score(X_test, y_test)

# Plot of a ROC curve for a specific class
plot_roc(fpr, tpr, roc_auc, subtype, method)

y_pred = xgb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

performance_df.loc[ 'B', method]['ROC-AUC'] = roc_auc
performance_df.loc['B', method]['Accuracy'] = score
performance_df.loc['B', method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])


ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Blues)



"""RF------------------------------------------------------------------------"""
# Random forest

method='RF'

# (rf, roc_auc_rf, rf_oob_score, score_rf, fpr_rf, tpr_rf) = random_forest(X_train, y_train, X_test, y_test, scale_pos_weight, n=500)
# #
# # # save model to file
# pickle.dump(rf, open("../OUTPUT/rf_{}.pickle.dat".format(subtype), "wb"))

# load model from file
rf = pickle.load(open("../OUTPUT/rf_{}.pickle.dat".format(subtype), "rb"))

y_pred_prob = rf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
score = rf.score(X_test, y_test)

plot_roc(fpr, tpr, roc_auc, subtype, method)

y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

performance_df.loc['B', method]['ROC-AUC'] = roc_auc
performance_df.loc['B', method]['Accuracy'] = score
performance_df.loc['B', method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])

ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Greens)
"""MLR-----------------------------------------------------------------------"""
# Logistic Regression
import Logistic_regression_modified as lr
method = 'MLR'
(alphas, ohe_normalized_alphas, e, ave_MSE) = logistic_regression(X_train, y_train, X, k=10)

score = 1 - ave_MSE

# alphas used for plotting ROC curve
alphas_plot = lr.logistic_regression(X_train, y_train)
alphas_plot.shape

y_pred_prob, y_pred = lr.LR_predict(alphas_plot, X_test)
X_test.drop('alpha_0', axis=1, inplace=True)

# y_pred_prob_xgb = xgb.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot of a ROC curve for a specific class
plot_roc(fpr, tpr, roc_auc, subtype, method='MLR')


cm = confusion_matrix(y_test, y_pred)

performance_df.loc['B', method]['ROC-AUC'] = roc_auc
performance_df.loc['B', method]['Accuracy'] = score
performance_df.loc['B', method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])

ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Greys)
"""SHAP----------------------------------------------------------------------"""
# Use SHAP to explain models
# %%
X_explain.shape
rf_explainer = shap.TreeExplainer(rf, data=None)
rf_shap_values = rf_explainer.shap_values(X_explain, approximate=True)
rf_shap_values = rf_shap_values[0] # For the random forest model, the shap TreeExplainer returns 2 sets of values, one for class 1 and one for class 0. They are symmetric, so you can use either.

# Creates the explainer based on the model.
xgb_explainer = shap.TreeExplainer(xgb, data=None)
# xgb_explainer = shap.TreeExplainer(xgb, data=X_explain, feature_dependence="independent")
xgb_shap_values = xgb_explainer.shap_values(X_explain, approximate=True)

with open('../OUTPUT/xgb_explainer_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(xgb_explainer, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/rf_explainer_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(rf_explainer, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/xgb_shap_values_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(xgb_shap_values, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/rf_shap_values_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(rf_shap_values, f, pickle.HIGHEST_PROTOCOL)

xgb_shap_values = pickle.load(open("../OUTPUT/xgb_shap_values_{}.pickle.dat".format(subtype), "rb"))

rf_shap_values = pickle.load(open("../OUTPUT/rf_shap_values_{}.pickle.dat".format(subtype), "rb"))


rf_shap_values_df = pd.DataFrame(rf_shap_values,
                                index=X_explain.index,
                                columns=X_explain.columns)

xgb_shap_values_df = pd.DataFrame(xgb_shap_values,
                                 index=X_explain.index,
                                 columns=X_explain.columns)

rf_shap_values_df = ohe_inverse(rf_shap_values_df)

xgb_shap_values_df = ohe_inverse(xgb_shap_values_df)

shap_values_CSV(xgb_shap_values_df, 'xgb', subtype)

shap_values_CSV(rf_shap_values_df, 'rf', subtype)

alphas.to_csv('../OUTPUT/shap_values_mlr_{}.csv'.format(subtype), index=True, header=['Score'])



"""RESULTS-------------------------------------------------------------------"""
# Plot resulting feature importances
# %%
plot_importances_genome(xgb_shap_values_df, rf_shap_values_df, alphas, subtype)

# Get importances values and genomic locations
(xgb_summary, rf_summary, sorted_alphas) = importance_summary(xgb_shap_values_df, rf_shap_values_df, alphas, subtype)


imp_merged = get_merged_results(xgb_summary, rf_summary, sorted_alphas, 30, subtype)

imp_merged.to_csv('../OUTPUT/{}_importances_merged_XGB_MLR.csv'.format(subtype), index=True, header=['Score'])
























"""
#######################################################################
MAIN subtype_C
#######################################################################
"""

# Data inmport
# %%
subtype = 'subtype_C'
pickle_ohe = '../!DATA/{}_ohe.pkl'.format(subtype)
pickle_seqdf = '../!DATA/{}.pkl'.format(subtype)

(seq_df, ohe_df) = get_data(pickle_seqdf, pickle_ohe)
seq_df.shape

ohe_df['Treatment'] = ohe_df['Treatment'].astype(int)

# Prepare data for training and testing
(X, y, X_train, X_test, y_train, y_test, scale_pos_weight) = get_train_test_split(ohe_df, test_size=0.1)



ohe_df_explain = ohe_df.groupby('Treatment', group_keys=False).apply(lambda x: x.sample(1000))
ohe_df_explain.shape
# Get only the ohe nucleotide info in X
X_explain = ohe_df_explain.drop(["Header","Subtype","Treatment"], axis=1)
# The target class is Ct_Group (high or low)
y_explain = ohe_df_explain["Treatment"]

# DataFrame to keep scores
# Run only in the beginning, then save at the end.
# index_names = [['02_AG', 'A', 'B', 'C'], ['XGB', 'RF', 'MLR']]
# multi_index = pd.MultiIndex.from_product(index_names, names=['Subtype', 'Method'])
# performance_df = pd.DataFrame(columns=['ROC-AUC', 'Accuracy', 'Precision'], index=multi_index)
# performance_df.loc['A', 'XGB']['Accuracy']

# %%
"""XGB-----------------------------------------------------------------------"""



# GRID SEARCH CV
################################################################################
params = {
        'subsample': [0.8],
        'colsample_bytree': [0.3],
        'max_depth': [50],
        'learning_rate': [0.01],
        'n_estimators': [500]
        }

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)

xgb = XGBClassifier(objective='binary:logistic', njobs=4, random_state=0, scale_pos_weight=scale_pos_weight)

grid = GridSearchCV(estimator=xgb,
                    param_grid=params,
                    scoring='roc_auc',
                    n_jobs=4,
                    cv=skf.split(X_train,y_train),
                    verbose=1)

grid.fit(X_train, y_train)

best_params = grid.best_params_
grid_results = pd.DataFrame(grid.cv_results_)
xgb = grid.best_estimator_

xgb = final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, best_params, subtype)

# save model to file
pickle.dump(xgb, open("../OUTPUT/xgb_{}.pickle.dat".format(subtype), "wb"))
################################################################################


method = 'XGB'

# load model from file
xgb = pickle.load(open("../OUTPUT/xgb_{}.pickle.dat".format(subtype), "rb"))

params = xgb.get_xgb_params()

# plot XGBoost training curves
xgb = final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, params, subtype)

# Prediction probabilities for ROC curve plotting
y_pred_prob = xgb.predict_proba(X_test)

# False positive rate, True positive rate, Decision threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

# ROC-AUC and Accuracy score
roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
score = xgb.score(X_test, y_test)

# Plot of a ROC curve for a specific class
plot_roc(fpr, tpr, roc_auc, subtype, method)

y_pred = xgb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

performance_df.loc['C', method]['ROC-AUC'] = roc_auc
performance_df.loc['C', method]['Accuracy'] = score
performance_df.loc['C', method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])


ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Blues)



"""RF------------------------------------------------------------------------"""
# Random forest

method='RF'

# (rf, roc_auc_rf, rf_oob_score, score_rf, fpr_rf, tpr_rf) = random_forest(X_train, y_train, X_test, y_test, scale_pos_weight, n=500)
# #
# # # save model to file
# pickle.dump(rf, open("../OUTPUT/rf_{}.pickle.dat".format(subtype), "wb"))

# load model from file
rf = pickle.load(open("../OUTPUT/rf_{}.pickle.dat".format(subtype), "rb"))

y_pred_prob = rf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
score = rf.score(X_test, y_test)

plot_roc(fpr, tpr, roc_auc, subtype, method)

y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

performance_df.loc['C', method]['ROC-AUC'] = roc_auc
performance_df.loc['C', method]['Accuracy'] = score
performance_df.loc['C', method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])

ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Greens)
"""MLR-----------------------------------------------------------------------"""
# Logistic Regression
import Logistic_regression_modified as lr
method = 'MLR'
(alphas, ohe_normalized_alphas, e, ave_MSE) = logistic_regression(X_train, y_train, X, k=10)

score = 1 - ave_MSE

# alphas used for plotting ROC curve
alphas_plot = lr.logistic_regression(X_train, y_train)
alphas_plot.shape

y_pred_prob, y_pred = lr.LR_predict(alphas_plot, X_test)
X_test.drop('alpha_0', axis=1, inplace=True)

# y_pred_prob_xgb = xgb.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot of a ROC curve for a specific class
plot_roc(fpr, tpr, roc_auc, subtype, method='MLR')


cm = confusion_matrix(y_test, y_pred)

performance_df.loc['C', method]['ROC-AUC'] = roc_auc
performance_df.loc['C', method]['Accuracy'] = score
performance_df.loc['C', method]['Precision'] = cm[1,1]/(cm[1,1] + cm[0,1])

ax = plot_confusion_matrix(y_test, y_pred,
                            ['Naive', 'Treated'],
                            subtype, method,
                            normalize=False,
                            cmap=plt.cm.Greys)
"""SHAP----------------------------------------------------------------------"""
# Use SHAP to explain models
# %%
X_explain.shape
rf_explainer = shap.TreeExplainer(rf, data=None)
rf_shap_values = rf_explainer.shap_values(X_explain, approximate=True)
rf_shap_values = rf_shap_values[0] # For the random forest model, the shap TreeExplainer returns 2 sets of values, one for class 1 and one for class 0. They are symmetric, so you can use either.

# Creates the explainer based on the model.
xgb_explainer = shap.TreeExplainer(xgb, data=None)
# xgb_explainer = shap.TreeExplainer(xgb, data=X_explain, feature_dependence="independent")
xgb_shap_values = xgb_explainer.shap_values(X_explain, approximate=True)

with open('../OUTPUT/xgb_explainer_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(xgb_explainer, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/rf_explainer_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(rf_explainer, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/xgb_shap_values_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(xgb_shap_values, f, pickle.HIGHEST_PROTOCOL)
with open('../OUTPUT/rf_shap_values_{}.pickle.dat'.format(subtype), 'wb') as f:
    # Pickle the 'data' using the highest protocol available.
    pickle.dump(rf_shap_values, f, pickle.HIGHEST_PROTOCOL)

xgb_shap_values = pickle.load(open("../OUTPUT/xgb_shap_values_{}.pickle.dat".format(subtype), "rb"))

rf_shap_values = pickle.load(open("../OUTPUT/rf_shap_values_{}.pickle.dat".format(subtype), "rb"))


rf_shap_values_df = pd.DataFrame(rf_shap_values,
                                index=X_explain.index,
                                columns=X_explain.columns)

xgb_shap_values_df = pd.DataFrame(xgb_shap_values,
                                 index=X_explain.index,
                                 columns=X_explain.columns)

rf_shap_values_df = ohe_inverse(rf_shap_values_df)

xgb_shap_values_df = ohe_inverse(xgb_shap_values_df)

shap_values_CSV(xgb_shap_values_df, 'xgb', subtype)

shap_values_CSV(rf_shap_values_df, 'rf', subtype)

alphas.to_csv('../OUTPUT/shap_values_mlr_{}.csv'.format(subtype), index=True, header=['Score'])



"""RESULTS-------------------------------------------------------------------"""
# Plot resulting feature importances
# %%
plot_importances_genome(xgb_shap_values_df, rf_shap_values_df, alphas, subtype)

# Get importances values and genomic locations
(xgb_summary, rf_summary, sorted_alphas) = importance_summary(xgb_shap_values_df, rf_shap_values_df, alphas, subtype)


imp_merged = get_merged_results(xgb_summary, rf_summary, sorted_alphas, 30, subtype)


imp_merged.to_csv('../OUTPUT/{}_importances_merged_XGB_MLR.csv'.format(subtype), index=True, header=['Score'])








performance_df.to_csv('../OUTPUT/PERFORMANCE_hiv_ML_models.csv', index=True)
