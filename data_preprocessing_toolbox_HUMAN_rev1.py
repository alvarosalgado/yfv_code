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

"""
#######################################################################
functions
#######################################################################
"""
def one_hot_encoding(seq_df, file_path='../DATA/Human_Analisys/DATA/'):

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


# Read YiBRA samples excel into a pd.DataFrame
file = '../DATA/Human_Analisys/DATA/2018-01_Salvador/CONSENSUS/YFV_Jan_2018_SampleList.xlsx'
metadata_excel = pd.read_excel(file, index_col='YiBRA_SSA_ID')

# Select only rows containing human samples
metadata1 = metadata_excel[metadata_excel['Host']=='Human']
metadata2 = metadata_excel[metadata_excel['Host']=='Human Serious or Fatal']
metadata = pd.concat([metadata1, metadata2], axis=0)

# Remove samples that were not sequenced
metadata = metadata[pd.notnull(metadata['YiBRA2_Library_Number'])]

metadata.shape
"""####################################"""


# check if there are only YFV samples
metadata['Original_Lab_Results'].unique()

# adjust nomenclature
metadata.loc[metadata['YiBRA2_Library_Number'] == 'library 4', 'YiBRA2_Library_Number'] = 'library4'

metadata.loc[metadata['YiBRA2_Library_Number'] == 'library 5', 'YiBRA2_Library_Number'] = 'library5'

metadata.loc[metadata['YiBRA2_Library_Number'] == 'library 6', 'YiBRA2_Library_Number'] = 'library6'

metadata.loc[metadata['YiBRA2_Library_Number'] == 'library 7', 'YiBRA2_Library_Number'] = 'library7'

"""####################################"""

# get all fasta files in a list
# I do this basically to get the library names so I can regex them and link the metadata. Now that i included more samples from different datasets that do not follow this library logic, i will do a brute force solution and create a list with all the library names and iterate over it.
# Also, I have to think the best way to create the dataframe columns, i think i will only need the id and class. Maybe not even the id, since i can use the index.
# file_list = glob.glob("../DATA/Human_Analisys/DATA/2018-01_Salvador/CONSENSUS/*.fasta")

library_list = ['library{}'.format(n) for n in range(1, 8)]
"""####################################"""

"""
FULL dataset
"""
filename = "../DATA/human_2.fasta"
# create sequence DataFrame
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(filename, "fasta")]
seqs = np.array([list(str(seq_rec.seq)) for seq_rec in SeqIO.parse(filename, "fasta")])

seqs.shape
seqs[1][0]
cols = list(range(seqs.shape[1]))
seq_df_yibra = pd.DataFrame(seqs, index=identifiers, columns=cols)
seq_df_yibra.insert(0, 'Library', np.nan)
seq_df_yibra.insert(1, 'BC', np.nan)
seq_df_yibra.insert(2, 'ID', np.nan)
seq_df_yibra.insert(3, 'Host', np.nan)
seq_df_yibra.insert(4, 'Class', np.nan)

# Parse fasta files to link sample metadata to DNA sequence
pattern_lib = 'library\d'
regex_lib = re.compile(pattern_lib)

pattern_bc = 'BC\d\d'
regex_bc = re.compile(pattern_bc)



for index, sample in seq_df_yibra.iterrows():
    library = regex_lib.search(str(index)).group()
    bc = regex_bc.search(str(index)).group()

    seq_df_yibra.loc[index, 'Library'] = library
    seq_df_yibra.loc[index, 'BC'] = bc

# Now go through metadata excel spreadsheet (in dataframe format)
pattern_nb = '(NB)(\d\d)'
regex_nb = re.compile(pattern_nb)
for index, sample in human_df.iterrows():
    # get the "library" info
    library = sample['YiBRA2_Library_Number']
    sample_NB = sample['YiBRA2_Barcode_Number']

    NB_number = regex_nb.search(sample_NB).group(2)
    barcode = 'BC'+NB_number

    seq_df_yibra.loc[(seq_df_yibra['Library'] == library) & (seq_df_yibra['BC'] == barcode), "ID"] = sample.name
    seq_df_yibra.loc[(seq_df_yibra['Library'] == library) & (seq_df_yibra['BC'] == barcode), "Host"] = sample['Host']


# Select only those that have IDs
seq_df_yibra = seq_df_yibra[pd.notnull(seq_df_yibra['ID'])]

# seq_df_original = seq_df.copy()
# seq_df_original.shape
#
# seq_df = seq_df_original.copy()



"""
Data Cleaning
"""







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

seq_df_original.to_pickle('../DATA/Human_Analisys/DATA/human_YFV_original_seq_df.pkl')
seq_ohe_df.to_pickle('../DATA/Human_Analisys/DATA/human_YFV_seq_ohe_df.pkl')
seq_df.to_pickle('../DATA/Human_Analisys/DATA/human_YFV_seq_df.pkl')
# Write report tables
host_count = seq_df.groupby('Host')["ID"].count()
host_count = host_count[["Human", "Human Serious or Fatal"]]
host_count.name = "Number of Sequences"

host_count.to_csv('./OUTPUT/HUMAN_sample_count.csv')
# table_human_count1_latex = host_count.to_latex()
# with open('./tables/table_human_count1_latex.txt', 'w') as f:
#     f.write(table_human_count1_latex)

seq_ohe_df.shape
