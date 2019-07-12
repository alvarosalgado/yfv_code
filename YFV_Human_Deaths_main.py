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
    X = ohe_df.drop(["Library", "BC", "ID", "Host", "Class"], axis=1)
    # The target class is Ct_Group (high or low)
    y = ohe_df["Class"]

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

    fig1.savefig('./figures/human_human_initial_xgb_model.png', format='png', dpi=300, transparent=False)

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
                        scoring='balanced_accuracy',
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

    fig1.savefig('./figures/human_final_xgb_model.png', format='png', dpi=300, transparent=False)

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
                                oob_score=True,
                                class_weight={0:1, 1:cw})

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
def logistic_regression(X_train, y_train, X, k=10):
    import Logistic_regression_modified as lr

    (e, ave_MSE, all_alphas, ave_alphas) = lr.LR_k_fold_CV(X_train, y_train, k)

    normalized_alphas = lr.normalize_alphas(ave_alphas, X)
    alphas = ohe_inverse_LR(normalized_alphas)

    return (alphas, e, ave_MSE)
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

def plot_importances_genome(xgb_shap_values_df, rf_shap_values_df, alphas):

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

    fig.savefig('./figures/human_importances_1.png', format='png', dpi=300, transparent=False)

"""
#######################################################################
Summary Plots
#######################################################################
"""
def importance_summary(xgb_shap_values_df, rf_shap_values_df, alphas):
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

    fig.savefig("./figures/human_fig_summary.png", format='png', dpi=300, transparent=False)

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
def get_merged_results(xgb_summary, rf_summary, sorted_alphas, top=10):
    xgb_summary_top = xgb_summary[:top]
    rf_summary_top = rf_summary[:top]
    sorted_alphas_top = sorted_alphas[:top]

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
    return results_all

"""
#######################################################################
Compare SNVs found by machine learning
Compare class 1 (high Ct) with class 0 (low Ct) and Alouatta samples.
Returns a table with the comparisons for the most important nucleotide positions found by the 3 ML algorithms
#######################################################################
"""
def validate_SNV(seq_df, imp_merged):

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

    seq_c1[:]='-'
    seq_c0[:]='-'

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

    # Creates a dataframe comparing the results
    df1 = pd.DataFrame(seq_c0).T
    df2 = pd.DataFrame(seq_c1).T

    table = pd.concat([df1, df2], axis=0)

    table = table.iloc[:, :15]

    table_latex = table.to_latex()

    return (table, table_latex)

"""
#######################################################################
MAIN
#######################################################################
"""
"""
#######################################################################
MAIN
#######################################################################
"""

# Data inmport
# %%
pickle_ohe = '../Human_Analisys/DATA/human_YFV_seq_ohe_df.pkl'
pickle_seqdf = '../Human_Analisys/DATA/human_YFV_seq_df.pkl'
pickle_seqdforiginal = '../Human_Analisys/DATA/human_YFV_original_seq_df.pkl'

(seq_df, ohe_df, seqdforiginal) = get_data(pickle_seqdf, pickle_ohe, pickle_seqdforiginal)

# Prepare data for training and testing
(X, y, X_train, X_test, y_train, y_test, scale_pos_weight) = get_train_test_split(ohe_df, test_size=0.1)

# Cell containing XGBoost Grid Search
# %%
# A parameter grid for XGBoost
# params = {
#         'subsample': [1.0],
#         'colsample_bytree': [0.3, 0.8],
#         'max_depth': [3, 5, 10],
#         'learning_rate': [0.001, 0.1, 1],
#         'n_estimators': [50, 250, 10000]
#         }
#
# (best_params, results) = grid_cv_xgb(X_train, y_train, scale_pos_weight, params, folds = 5)
#
# results["mean_test_score"].unique()
# all parameter combinations resulted in the same score...

# params_series = results.loc[results['mean_test_score'] == np.max(results['mean_test_score']), 'params']
# for p in params_series:
#     print(p)
#
# best_params = params_series.iloc[2]


# Train models
# %%
best_params = {'colsample_bytree': 0.3,
               'learning_rate': 0.001,
               'max_depth': 3,
               'n_estimators': 50,
               'subsample': 1.0}

xgb = final_xgb(X_train, y_train, X_test, y_test, scale_pos_weight, best_params)

xgb.score(X, y)

(rf, roc_auc, rf.oob_score_) = random_forest(X_train, y_train, X_test, y_test, scale_pos_weight, n=100)
rf.oob_score_

(alphas, e, ave_MSE) = logistic_regression(X_train, y_train, X, k=10)
1-ave_MSE

# Use SHAP to explain models
# %%
(xgb_explainer, rf_explainer, xgb_shap_values, rf_shap_values) = get_explainer(xgb, rf, X_train)

xgb_shap_values.shape
X_train.columns
rf_shap_values_df = pd.DataFrame(rf_shap_values,
                                index=X_train.index,
                                columns=X_train.columns)

xgb_shap_values_df = pd.DataFrame(xgb_shap_values,
                                 index=X_train.index,
                                 columns=X_train.columns)

rf_shap_values_df = ohe_inverse(rf_shap_values_df)
xgb_shap_values_df = ohe_inverse(xgb_shap_values_df)

"""
#######################################################################
Plot results
#######################################################################
"""
# Plot resulting feature importances
# %%
plot_importances_genome(xgb_shap_values_df, rf_shap_values_df, alphas)

# Get importances values and genomic locations
(xgb_summary, rf_summary, sorted_alphas) = importance_summary(xgb_shap_values_df, rf_shap_values_df, alphas)

"""
#######################################################################
Analyze results
#######################################################################
"""

imp_merged = get_merged_results(xgb_summary, rf_summary, sorted_alphas, 30)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

ax.barh(imp_merged.index[0:30], imp_merged[0:30])
ax.set_yticks(imp_merged.index[0:30])
ax.set_yticklabels(imp_merged.index[0:30])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature Importance')
ax.set_title('XGBoosting')
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

fig.tight_layout();

fig.savefig("./figures/human_combined_30_summary.png", format='png', dpi=300, transparent=False)

# The analysis below and the results shown in "table" demonstrate the power of XGBoost. It only picked 3 features, and there was a total of 5 that really had any informative value. All the rest, that both RF and LR gave some importance (albeit small), have no information at all, given that they do not contain a nucleotide that is different from the most frequent one in the other class and in the Alouatta samples.
(table, table_latex) = validate_SNV(seq_df, imp_merged)

# Requires usepackage{booktabs} on LaTex document
with open('./tables/human_table1_latex.txt', 'w') as f:
    f.write(table_latex)


table_clean = table.copy()

for col in table_clean.columns:
    if '-' in table_clean.loc['Serious', col]:
        table_clean.drop(col, axis=1, inplace=True)

table2_latex = table_clean.to_latex()
with open('./tables/human_table2_latex.txt', 'w') as f:
    f.write(table2_latex)


snv_to_analyze = table_clean.loc['Serious', :]



import ref_genome_polyprot_toolbox as g_tool

dataset_file = '../Callithrix_Analysis/DATA/!CLEAN/ALL_YFV.aln'
ref_genome_file = '../Callithrix_Analysis/DATA/!CLEAN/YFV_BeH655417_JF912190.gb'
ref_polyprot_file = '../Callithrix_Analysis/DATA/!CLEAN/YFV_polyprotein_AFH35044.gp'

(ref_genome, ref_polyprot, seq) = g_tool.read_data(ref_genome_file, ref_polyprot_file, dataset_file)

querry_set = SeqIO.parse('../Human_Analisys/DATA/2018-01_Salvador/CONSENSUS/YiBRAlibrary1_30012018.consensus.fasta', "fasta")
first_record = next(querry_set)
second_record = next(querry_set)
seq = second_record.seq

seq_rel_start = g_tool.find_align_start(seq, ref_genome, 20)
dic_prot = g_tool.read_polyprotein(ref_polyprot)

report_dic = {}
for nn_pos, nn in snv_to_analyze.iteritems():

    dic = {}
    # nn_pos = snv_to_analyze.index[1]
    # nn = snv_to_analyze[nn_pos]
    nn_pos = int(nn_pos)

    (aa_pos, aa, codon, codon_pos) = g_tool.pos_aminoacid(nn_pos, seq_rel_start, ref_genome, ref_polyprot)

    prot = g_tool.which_protein(aa_pos, dic_prot)

    (codon_seq, aa_seq, ref_pos, codon_ref, aa_ref, codon_pos) = g_tool.seq_snv_info(nn_pos, seq, ref_genome, ref_polyprot)

    codon_seq = list(codon_ref)
    codon_seq[codon_pos] = nn
    codon_seq = Seq("".join(codon_seq))
    aa_seq = codon_seq.translate()

    dic["protein"] = str(prot)
    dic["Reference nn pos"] = str(ref_pos)
    dic["Reference codon"] = str(codon_ref)
    dic["Reference aa"] = str(aa_ref)
    dic["Sequence codon"] = str(codon_seq)
    dic["Sequence aa"] = str(aa_seq)
    dic["Codon position (0, 1, 2)"] = str(codon_pos)

    df = pd.DataFrame(dic, index=[nn_pos])
    report_dic[nn_pos] = df

table3 = pd.concat(list(report_dic.values()))
table3_latex = table3.to_latex()
with open('./tables/human_table3_latex.txt', 'w') as f:
    f.write(table3_latex)
