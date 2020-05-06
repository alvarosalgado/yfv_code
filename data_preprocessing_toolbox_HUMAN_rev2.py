#!/usr/bin/env python3
"""
#######################################################################

- Specific for "YFV Human Severity investigation"

- Read the fasta files containing the sequences' nucleotides and the excel files containing their metadata.
- Clean the data.
- Turn it into "one-hot encoded" data.
- Save it to csv and pkl.

- This is the same as 'data_preprocessing_YFV_rev1.ipynb', except for the markdown
#######################################################################
"""

from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Alphabet import IUPAC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("whitegrid")

import re
import glob
import datetime
import time
import progressbar

import os, sys
cwd = os.getcwd()

"""
#######################################################################
functions
#######################################################################
"""
# This function got so small there's no need for it to be a function anymore.



# def one_hot_encoding(seq_df):
#     '''Codify categorical nucleotide data into numeric one hot encoded'''
#
#     seq_ohe_df = pd.get_dummies(seq_df)
#     seq_ohe_df.apply(pd.to_numeric)
#
#     return seq_ohe_df


""" /////////////////////////////////////////////////////////////////////// """

""" SET THE STAGE... """

# Create OUTPUT dir inside DATA dir, where all processed data, figures, tbles, ect will be stored

working_dir = '/Users/alvarosalgado/Google Drive/Bioinformática/!Qualificação_alvaro/YFV'

if os.path.isdir(working_dir+'/2_OUTPUT/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/')

if os.path.isdir(working_dir+'/2_OUTPUT/HUMAN/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/HUMAN/')

if os.path.isdir(working_dir+'/2_OUTPUT/HUMAN/FIGURES/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/HUMAN/FIGURES/')
if os.path.isdir(working_dir+'/2_OUTPUT/HUMAN/TABLES/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/HUMAN/TABLES/')
if os.path.isdir(working_dir+'/2_OUTPUT/HUMAN/PICKLE/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/HUMAN/PICKLE/')

out_dir = working_dir+'/2_OUTPUT/HUMAN'
fig_dir = working_dir+'/2_OUTPUT/HUMAN/FIGURES'
tab_dir = working_dir+'/2_OUTPUT/HUMAN/TABLES'
pik_dir = working_dir+'/2_OUTPUT/HUMAN/PICKLE'
data_dir = working_dir+'/1_DATA/Human_Analisys'

log_file = out_dir+'/LOG_preprocessing_{0}.txt'.format(datetime.datetime.now())
with open(log_file, 'w') as log:
    x = datetime.datetime.now()
    log.write('LOG file for data preprocessing\n{0}\n\n'.format(x))



""" /////////////////////////////////////////////////////////////////////// """
""" /////////////////////////////////////////////////////////////////////// """
""" /////////////////////////////////////////////////////////////////////// """
""" /////////////////////////////////////////////////////////////////////// """

#%%
""" MAIN DATASET (all together) """
# Open the fasta file
fasta_file = data_dir+'/ALN_YFV_ALL_HUMAN+REF.fasta'
# create sequence DataFrame
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(fasta_file, "fasta")]
seqs = np.array([list(str(seq_rec.seq.lower())) for seq_rec in SeqIO.parse(fasta_file, "fasta")])
seqs.shape
cols = np.array(range(seqs.shape[1]))
cols = cols + 1


# I will work on separate DataFrames for the data (nucleotides) and metadata. In the end, after data cleaning, I will merge them.
# Sequence df
ALL_seq_df = pd.DataFrame(seqs, index=identifiers, columns=cols)

# Metadata Df
meta_columns = ['Library', 'BC', 'ID', 'Host', 'Class', 'Dataset']
ALL_meta_df = pd.DataFrame(data=None, columns=meta_columns, index=identifiers)
ALL_meta_df.shape


"""
#######################################################################
Yibra Dataset
#######################################################################
"""
# Read YiBRA metadata samples excel into a pd.DataFrame
meta_file_yibra = data_dir+'/YFV_Jan_2018_SampleList.xlsx'
meta_yibra_xls_df = pd.read_excel(meta_file_yibra, index_col='YiBRA_SSA_ID')
meta_yibra_xls_df['Host'].unique()

# Select only rows containing human samples
metadata1 = meta_yibra_xls_df[meta_yibra_xls_df['Host']=='Human']
metadata2 = meta_yibra_xls_df[meta_yibra_xls_df['Host']=='Human Serious or Fatal']
metadata3 = meta_yibra_xls_df[meta_yibra_xls_df['Host']=='Human Grave']

meta_yibra_xls_df = pd.concat([metadata1, metadata2, metadata3], axis=0)

# Remove samples that were not sequenced
meta_yibra_xls_df = meta_yibra_xls_df[pd.notnull(meta_yibra_xls_df['YiBRA2_Library_Number'])]

meta_yibra_xls_df.shape
"""####################################"""


# check if there are only YFV samples
meta_yibra_xls_df['Original_Lab_Results'].unique()
meta_yibra_xls_df['Host'].unique()

# adjust nomenclature
meta_yibra_xls_df.loc[meta_yibra_xls_df['YiBRA2_Library_Number'] == 'library 4', 'YiBRA2_Library_Number'] = 'library4'

meta_yibra_xls_df.loc[meta_yibra_xls_df['YiBRA2_Library_Number'] == 'library 5', 'YiBRA2_Library_Number'] = 'library5'

meta_yibra_xls_df.loc[meta_yibra_xls_df['YiBRA2_Library_Number'] == 'library 6', 'YiBRA2_Library_Number'] = 'library6'

meta_yibra_xls_df.loc[meta_yibra_xls_df['YiBRA2_Library_Number'] == 'library 7', 'YiBRA2_Library_Number'] = 'library7'

# get all fasta files in a list
# I do this basically to get the library names so I can regex them and link the metadata. Now that i included more samples from different datasets that do not follow this library logic, i will do a brute force solution and create a list with all the library names and iterate over it.
# Also, I have to think the best way to create the dataframe columns, i think i will only need the id and class. Maybe not even the id, since i can use the index.
# file_list = glob.glob("../DATA/Human_Analisys/DATA/2018-01_Salvador/CONSENSUS/*.fasta")

library_list = ['library{}'.format(n) for n in range(1, 8)]

#%%
"""Fasta file"""
# Open the fasta file
fasta_file_yibra = data_dir+'/Yibra.fasta'
# create sequence DataFrame
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(fasta_file_yibra, "fasta")]



# I will work on separate DataFrames for the data (nucleotides) and metadata. In the end, after data cleaning, I will merge them.
# Sequence df
seq_df_yibra = ALL_seq_df.loc[identifiers, :].copy()

# Metadata Df
meta_columns = ['Library', 'BC', 'ID', 'Host', 'Class', 'Dataset']
meta_df_yibra = pd.DataFrame(data=None, columns=meta_columns, index=identifiers)
meta_df_yibra.shape
seq_df_yibra.shape
# seq_df_yibra.insert(0, 'Library', 'library')
# seq_df_yibra.insert(1, 'BC', 'bc')
# seq_df_yibra.insert(2, 'ID', 'id')
# seq_df_yibra.insert(3, 'Host', 'host')
# seq_df_yibra.insert(4, 'Class', 'class')
# seq_df_yibra.insert(5, 'Dataset', 'dataset')

# Custom made code to deal especifically with YIBRA dataset, which has "library" and "barcode" as its identifiers in the index (in the fasta ID), which I'll uso to link the metadata spreadsheet.
#Parse fasta files to link sample metadata to DNA sequence
# First, I will read the meta_df_yibra index, which contains the same index as the sequence dataframe, and extract the info regarding Library and Barcode Number
pattern_lib = 'library\d'
regex_lib = re.compile(pattern_lib)

pattern_bc = 'BC\d\d'
regex_bc = re.compile(pattern_bc)


for index, sample in meta_df_yibra.iterrows():
    library = "empty"
    bc = "empty"
    if regex_lib.search(str(index)):
        library = regex_lib.search(str(index)).group()
    if regex_bc.search(str(index)):
        bc = regex_bc.search(str(index)).group()

    meta_df_yibra.loc[index, 'Library'] = library
    meta_df_yibra.loc[index, 'BC'] = bc


# Now go through metadata excel spreadsheet (in dataframe format)
# meta_yibra_xls_df is the excel
# meta_df_yibra is the metadata DataFrame I am filling

pattern_nb = '(NB)(\d\d)'
regex_nb = re.compile(pattern_nb)
for index, sample in meta_yibra_xls_df.iterrows():
    # get the "library" info
    library = sample['YiBRA2_Library_Number']
    sample_NB = sample['YiBRA2_Barcode_Number']

    NB_number = regex_nb.search(sample_NB).group(2)
    barcode = 'BC'+NB_number

    if (library in meta_df_yibra['Library'].unique()) and (barcode in meta_df_yibra['BC'].unique()):
        meta_df_yibra.loc[(meta_df_yibra['Library'] == library) & (meta_df_yibra['BC'] == barcode), "ID"] = sample.name
        meta_df_yibra.loc[(meta_df_yibra['Library'] == library) & (meta_df_yibra['BC'] == barcode), "Host"] = sample['Host']

# Select only those that have IDs
meta_df_yibra = meta_df_yibra[meta_df_yibra['ID'].isnull() == False]
meta_df_yibra.shape
# set the label, between fatal and non fatal
meta_df_yibra.loc[:, 'Class'] = 0
meta_df_yibra.loc[meta_df_yibra['Host'] == 'Human Serious or Fatal', 'Class'] = 1
meta_df_yibra.loc[:, 'Dataset'] = 'Yibra'

seq_df_yibra.shape
meta_df_yibra.shape

seq_df_yibra = seq_df_yibra.loc[meta_df_yibra.index, :]
seq_df_yibra.shape

# df = meta_df_yibra.merge(seq_df_yibra, left_index=True, right_index=True)
# df.shape
'''
I think I'll have to work on the separate datasets first.
Now that I have them all alligned, I will separate the 4 datasets on AliView before importing.
'''

"""
#######################################################################
Marielton Dataset
#######################################################################
"""

# Open the fasta file
fasta_file_marielton = data_dir+"/Marielton.fasta"
# create sequence DataFrame
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(fasta_file_marielton, "fasta")]
# seqs = np.array([list(str(seq_rec.seq)) for seq_rec in SeqIO.parse(fasta_file_marielton, "fasta")])
#
# seqs.shape
# cols = np.array(range(seqs.shape[1]))
# cols = cols + 1

seq_df_mari = ALL_seq_df.loc[identifiers, :].copy()

meta_columns = ['Library', 'BC', 'ID', 'Host', 'Class', 'Dataset']
meta_df_mari = pd.DataFrame(data=None, columns=meta_columns, index=identifiers)
# seq_df_mari.insert(0, 'Library', 'library')
# seq_df_mari.insert(1, 'BC', 'bc')
# seq_df_mari.insert(2, 'ID', 'id')
# seq_df_mari.insert(3, 'Host', 'host')
# seq_df_mari.insert(4, 'Class', 'class')
# seq_df_mari.insert(5, 'Dataset', 'dataset')

meta_df_mari.loc[:, 'Class'] = 1
meta_df_mari.loc[:, 'Host'] = 'Human'
meta_df_mari['ID'] = meta_df_mari.index
meta_df_mari.loc[:, 'Dataset'] = 'Marielton'

seq_df_mari.shape
meta_df_mari.shape

"""
#######################################################################
Talita Cura Dataset
#######################################################################
"""

# Open the fasta file
fasta_file_tc = data_dir+"/Talita_cura.fasta"
# create sequence DataFrame
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(fasta_file_tc, "fasta")]
# seqs = np.array([list(str(seq_rec.seq)) for seq_rec in SeqIO.parse(fasta_file_tc, "fasta")])
#
# seqs.shape
# cols = np.array(range(seqs.shape[1]))
# cols = cols + 1
#
# seq_df_tcura = pd.DataFrame(seqs, index=identifiers, columns=cols)
seq_df_tcura = ALL_seq_df.loc[identifiers, :].copy()

meta_columns = ['Library', 'BC', 'ID', 'Host', 'Class', 'Dataset']
meta_df_tcura = pd.DataFrame(data=None, columns=meta_columns, index=identifiers)
# seq_df_tcura.insert(0, 'Library', 'library')
# seq_df_tcura.insert(1, 'BC', 'bc')
# seq_df_tcura.insert(2, 'ID', 'id')
# seq_df_tcura.insert(3, 'Host', 'host')
# seq_df_tcura.insert(4, 'Class', 'class')
# seq_df_tcura.insert(5, 'Dataset', 'dataset')

meta_df_tcura.loc[:, 'Class'] = 0
meta_df_tcura.loc[:, 'Host'] = 'Human'
meta_df_tcura['ID'] = meta_df_tcura.index
meta_df_tcura.loc[:, 'Dataset'] = 'T_cura'

seq_df_tcura.shape
meta_df_tcura.shape


"""
#######################################################################
Talita Obitos Dataset
#######################################################################
"""

# Open the fasta file
fasta_file_to = data_dir+"/Talita_obitos.fasta"
# create sequence DataFrame
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(fasta_file_to, "fasta")]
# seqs = np.array([list(str(seq_rec.seq)) for seq_rec in SeqIO.parse(fasta_file_to, "fasta")])
#
# seqs.shape
# cols = np.array(range(seqs.shape[1]))
# cols = cols + 1

# seq_df_tob = pd.DataFrame(seqs, index=identifiers, columns=cols)
seq_df_tob = ALL_seq_df.loc[identifiers, :].copy()

meta_columns = ['Library', 'BC', 'ID', 'Host', 'Class', 'Dataset']
meta_df_tob = pd.DataFrame(data=None, columns=meta_columns, index=identifiers)

# seq_df_tob.insert(0, 'Library', 'library')
# seq_df_tob.insert(1, 'BC', 'bc')
# seq_df_tob.insert(2, 'ID', 'id')
# seq_df_tob.insert(3, 'Host', 'host')
# seq_df_tob.insert(4, 'Class', 'class')
# seq_df_tob.insert(5, 'Dataset', 'dataset')

meta_df_tob.loc[:, 'Class'] = 1
meta_df_tob.loc[:, 'Host'] = 'Human'
meta_df_tob['ID'] = meta_df_tob.index
meta_df_tob.loc[:, 'Dataset'] = 'T_obitos'

seq_df_tob.shape
meta_df_tob.shape

#%%
"""
Data Cleaning

Here I need to choose which datasets I'll use. Since Talita's dataset might not be available to me, I'll first use it only as validation.
"""

# dataframes = [seq_df_yibra, seq_df_mari]
dataframes = [seq_df_yibra, seq_df_mari, seq_df_tcura, seq_df_tob]

meta = [meta_df_yibra, meta_df_mari, meta_df_tcura, meta_df_tob]


# seq_df_yibra.shape
# meta_df_tob.shape

#dataframes = [seq_df_yibra]

seq_df = pd.concat(dataframes)
seq_df_original = seq_df.copy()

meta_df = pd.concat(meta)
meta_df.groupby(['Class']).count()

with open(log_file, 'a') as log:
    x = datetime.datetime.now()
    log.write('{2}\nWe initially have:\n\t{0} samples\n\t{1} nucleotides\n\n'.format(seq_df.shape[0],seq_df.shape[1],x))



"""
Data Cleaning
"""
#%%
meta_df.groupby("Dataset").count()
n_mari = meta_df.groupby("Dataset").count().loc['Marielton', 'ID']
n_yibr = meta_df.groupby("Dataset").count().loc['Yibra', 'ID']
n_tc = meta_df.groupby("Dataset").count().loc['T_cura', 'ID']
n_to = meta_df.groupby("Dataset").count().loc['T_obitos', 'ID']

with open(log_file, 'a') as log:
    x = datetime.datetime.now()
    log.write('{4}\nInitial distribution:\n\t{0} Marielton sequences\n\t{1} Yibra sequences\n\t{2} Talita Cura sequences\n\t{3} Talita Obitos sequences\n\n'.format(n_mari, n_yibr, n_tc, n_to, x))


#%%
# Replace non sequenced positions with np.nan
unique = pd.unique(seq_df.values.ravel('K'))
# Get a list of all characters other than "actg" are present in the DataFrame
unique = unique[unique!='a']
unique = unique[unique!='c']
unique = unique[unique!='t']
unique = unique[unique!='g']
to_replace = unique

# Replace them with np.nan
seq_df.replace(to_replace, np.nan, inplace=True)

# Remove all sequences that contain more than 10% gaps or "N".
threshold = int((seq_df.shape[1])*0.9)
seq_df.dropna(axis=0, how='any', thresh=threshold, inplace=True)

with open(log_file, 'a') as log:
    x = datetime.datetime.now()
    log.write('{2}\nAfter removing samples with more than 10% empty positions, we have\n\t{0} samples\n\t{1} nucleotides\n\n'.format(seq_df.shape[0], seq_df.shape[1], x))


# seq_df.groupby("Dataset").count()
# n_mari = seq_df.groupby("Dataset").count().iloc[0,0]
# n_yibr = seq_df.groupby("Dataset").count().iloc[3,0]
# n_tc = seq_df.groupby("Dataset").count().iloc[1,0]
# n_to = seq_df.groupby("Dataset").count().iloc[2,0]
#
# with open(log_file, 'a') as log:
#     x = datetime.datetime.now()
#     log.write('{4}\nNew distribution:\n\t{0} Marielton sequences\n\t{1} Yibra sequences\n\t{2} Talita Cura sequences\n\t{3} Talita Obitos sequences\n\n'.format(n_mari, n_yibr, n_tc, n_to, x))


# Removes all columns (positions) that contain any gap or "N"
seq_df.dropna(axis=1, how='any', inplace=True)

with open(log_file, 'a') as log:
    x = datetime.datetime.now()
    log.write('{2}\nAfter dropping ALL columns with ANY empty positions, we have:\n\t{0} samples\n\t{1} nucleotides\n\n'.format(seq_df.shape[0], seq_df.shape[1], x))


# Removes all columns that do not contain variation in the nucleotide, i.e., that are the same for all sequences.



for col in seq_df.columns:
    if seq_df[col].nunique() == 1:
        seq_df.drop(col, axis=1, inplace=True)

with open(log_file, 'a') as log:
    x = datetime.datetime.now()
    log.write('{2}\nAfter removing columns in which there is no variation, we have:\n\t{0} samples\n\t{1} nucleotides\n\n'.format(seq_df.shape[0], seq_df.shape[1], x))







#%%
'''Codify categorical nucleotide data into numeric one hot encoded'''

ohe_df = pd.get_dummies(seq_df)
ohe_df = ohe_df.apply(pd.to_numeric)

ohe_df.shape
seq_df.shape
meta_df.shape
#%%
# Merge dataset with metadata
seq_ohe_df = meta_df.merge(ohe_df, left_index=True, right_index=True)
seq_ohe_df.shape
seq_ohe_df.groupby('Dataset').count()

#%%
# How many sequences of each group are we left with?
class0 = seq_ohe_df.groupby('Class').count().loc[0,'ID']
class1 = seq_ohe_df.groupby('Class').count().loc[1,'ID']

with open(log_file, 'a') as log:
    x = datetime.datetime.now()
    log.write('{2}\nWe are left with:\n\t{0} non-severe\n\t{1} severe/fatal'.format(class0, class1, x))

print("We are left with:\n\t{0} non-severe\n\t{1} severe/fatal".format(class0, class1))
#%%
seq_df_original.to_pickle(pik_dir+'/human_YFV_original_seq_df.pkl')
seq_ohe_df.to_pickle(pik_dir+'/human_YFV_seq_ohe_df.pkl')
seq_df.to_pickle(pik_dir+'/human_YFV_seq_df.pkl')






# meta_df[meta_df['ID'].isnull()]
# seq_ohe_df.to_csv(file_path + filename + '_ohe.csv', index=True, header=True, decimal='.', sep=',', float_format='%.2f')
#
# seq_ohe_df.to_pickle(file_path + filename + '_ohe.pkl')
#
# seq_df.to_pickle(file_path + filename + '_dataframe.pkl')
#
# seq_df_original.to_pickle(file_path + filename + '_original.pkl')

# Write report tables
# host_count = seq_df.groupby('Host')["ID"].count()
# host_count = host_count[["Human", "Human Serious or Fatal"]]
# host_count.name = "Number of Sequences"
#
# host_count.to_csv('./OUTPUT/HUMAN_sample_count.csv')
# table_human_count1_latex = host_count.to_latex()
# with open('./tables/table_human_count1_latex.txt', 'w') as f:
#     f.write(table_human_count1_latex)

seq_ohe_df.shape
seq_df_original.shape
seq_df.shape
