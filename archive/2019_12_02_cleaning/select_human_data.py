#!/usr/bin/env python3

from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Alphabet import IUPAC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("whitegrid")

import re
import glob

"""
#######################################################################
functions
#######################################################################
"""
def one_hot_encoding(seq_df, file_path='../Human_Analisys/DATA/'):

    nucleotides_df = seq_df.iloc[:, 5:]
    seq_ohe_df = pd.get_dummies(nucleotides_df)

    seq_ohe_df.insert(0, 'Library', seq_df['Library'])
    seq_ohe_df.insert(1, 'BC', seq_df['BC'])
    seq_ohe_df.insert(2, 'ID', seq_df['ID'])
    seq_ohe_df.insert(3, 'Host', seq_df['Host'])
    seq_ohe_df.insert(4, 'Class', seq_df['Class'])

    seq_ohe_df.to_csv(file_path + 'human_YFV_seq_ohe_df.csv', index=True, header=True, decimal='.', sep=',', float_format='%.2f')

    seq_ohe_df.to_pickle(file_path + 'human_YFV_seq_ohe_df.pkl')

    seq_df.to_pickle(file_path + 'human_YFV_seq_df.pkl')

    return seq_ohe_df


# Read excel into a pd.DataFrame
file = '../Human_Analisys/DATA/2018-01_Salvador/CONSENSUS/YFV_Jan_2018_SampleList.xlsx'
sample_list_excel = pd.read_excel(file, index_col='YiBRA_SSA_ID')

# Select only rows containing human samples
human_df1 = sample_list_excel[sample_list_excel['Host']=='Human']
human_df2 = sample_list_excel[sample_list_excel['Host']=='Human Serious or Fatal']
human_df = pd.concat([human_df1, human_df2], axis=0)

# Remove samples that were not sequenced
human_df = human_df[pd.notnull(human_df['YiBRA2_Library_Number'])]

"""####################################"""


# check if there are only YFV samples
human_df['Original_Lab_Results'].unique()

human_df.loc[human_df['YiBRA2_Library_Number'] == 'library 4', 'YiBRA2_Library_Number'] = 'library4'

human_df.loc[human_df['YiBRA2_Library_Number'] == 'library 5', 'YiBRA2_Library_Number'] = 'library5'

human_df.loc[human_df['YiBRA2_Library_Number'] == 'library 6', 'YiBRA2_Library_Number'] = 'library6'

human_df.loc[human_df['YiBRA2_Library_Number'] == 'library 7', 'YiBRA2_Library_Number'] = 'library7'

"""####################################"""

# get all fasta files in a list
file_list = glob.glob("../Human_Analisys/DATA/2018-01_Salvador/CONSENSUS/*.fasta")

"""####################################"""

# Parse fasta files to link sample metadata to DNA sequence
pattern1 = 'library\d'
regex1 = re.compile(pattern1)

seq_dict = {}
for filename in file_list:
    # Get file's library id
    library = regex1.search(filename).group()

    # create sequence DataFrame
    identifiers = [seq_rec.id for seq_rec in SeqIO.parse(filename, "fasta")]
    seqs = np.array([list(str(seq_rec.seq)) for seq_rec in SeqIO.parse(filename, "fasta")])
    cols = list(range(seqs.shape[1]))
    seq_df = pd.DataFrame(seqs, index=identifiers, columns=cols)

    seq_df.insert(0, 'Library', np.nan)
    seq_df.insert(1, 'BC', np.nan)
    seq_df.insert(2, 'ID', np.nan)
    seq_df.insert(3, 'Host', np.nan)
    seq_df.insert(4, 'Class', np.nan)

    pattern_bc = 'BC\d\d'
    regex_bc = re.compile(pattern_bc)

    for index, sample in seq_df.iterrows():
        bc = regex_bc.search(str(index)).group()
        seq_df.loc[index, 'BC'] = bc

    seq_df['Library'] = library

    seq_dict[library] = seq_df

    # print(library)
    # print(identifiers)

# Now I have all sequences in dataframes that contain the "library" and "barcode" information.
seq_dict['library2'].head(12)
# Link the metadata in human_df to the sequences
human_df.head()

for index, sample in human_df.iterrows():
    # print(index)
    library = sample['YiBRA2_Library_Number']

    pattern = '(NB)(\d\d)'
    regex = re.compile(pattern)
    sample_NB = sample['YiBRA2_Barcode_Number']
    NB_number = regex.search(sample_NB).group(2)

    barcode = 'BC'+NB_number

    seq_df = seq_dict[library]

    seq_df.loc[seq_df['BC'] == barcode, 'ID'] = sample.name
    seq_df.loc[seq_df['BC'] == barcode, 'Host'] = sample['Host']

seq_dict['library2'].head(12)

list_seq_df = []
for key in seq_dict:
    list_seq_df.append(seq_dict[key])

seq_df = pd.concat(list_seq_df)

seq_df = seq_df[pd.notnull(seq_df['ID'])]

seq_df[seq_df['ID'] == 'SA33']


"""
Data Cleaning
"""

seq_df['Class'] = 0
seq_df.loc[seq_df['Host'] == 'Human Serious or Fatal', 'Class'] = 1
seq_df.replace('N', np.nan, inplace=True)
seq_df.replace('-', np.nan, inplace=True)

"""--------------------------------------------------------------------------"""
"""
Below I analyze the quality of the only "fatal" sample sequenced.
It has a coverage of only 73%, not suitable for our analysis.
"""
sa8 = seq_df[seq_df["ID"] == "SA8"]
valid = sa8.iloc[0, 5:].count()
len = sa8.shape[1]

cover = valid/len
"""--------------------------------------------------------------------------"""

# Remove all sequences that contain more than 10% gaps or "N".
threshold = int(seq_df.shape[1]*0.9)
seq_df.dropna(axis=0, how='any', thresh=threshold, inplace=True)
seq_df.shape

# Removes all columns (positions) that contain any gap or "N"
seq_df.dropna(axis=1, how='any', inplace=True)

# Removes all columns that do not contain variation in the nucleotide, i.e., that are the same for all sequences.
for col in seq_df.columns:
    if seq_df[col].nunique() == 1:
        seq_df.drop(col, axis=1, inplace=True)

# How many sequences of each group are we left with?
seq_df.groupby('Host').count()
# we are left with 17 normal human cases and 4 serious human cases

# check if these 4 cases are deaths or serious.
seq_df[seq_df["Host"] == "Human Serious or Fatal"]
seq_ohe_df = one_hot_encoding(seq_df)


# Write report tables
host_count = seq_df.groupby('Host')["ID"].count()
host_count = host_count[["Human", "Human Serious or Fatal"]]
host_count.name = "Number of Sequences"
table_human_count1_latex = host_count.to_latex()
with open('./tables/table_human_count1_latex.txt', 'w') as f:
    f.write(table_human_count1_latex)
