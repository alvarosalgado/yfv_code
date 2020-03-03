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

    nucleotides_df = seq_df.iloc[:, 6:]
    metadata_df = seq_df.iloc[:, :6]

    seq_ohe_df = pd.get_dummies(nucleotides_df)

    dataframes = [metadata_df, seq_ohe_df]
    seq_ohe_df = pd.concat(dataframes, axis=1)

    seq_ohe_df.to_csv(file_path + 'human_YFV_seq_ohe_df.csv', index=True, header=True, decimal='.', sep=',', float_format='%.2f')

    seq_ohe_df.to_pickle(file_path + 'human_YFV_seq_ohe_df.pkl')

    seq_df.to_pickle(file_path + 'human_YFV_seq_df.pkl')

    return seq_ohe_df


"""
#######################################################################
Yibra Dataset
#######################################################################
"""
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

# get all fasta files in a list
# I do this basically to get the library names so I can regex them and link the metadata. Now that i included more samples from different datasets that do not follow this library logic, i will do a brute force solution and create a list with all the library names and iterate over it.
# Also, I have to think the best way to create the dataframe columns, i think i will only need the id and class. Maybe not even the id, since i can use the index.
# file_list = glob.glob("../DATA/Human_Analisys/DATA/2018-01_Salvador/CONSENSUS/*.fasta")

library_list = ['library{}'.format(n) for n in range(1, 8)]

# Open the fasta file
filename = "../DATA/Yibra.fasta"
# create sequence DataFrame
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(filename, "fasta")]
seqs = np.array([list(str(seq_rec.seq)) for seq_rec in SeqIO.parse(filename, "fasta")])

seqs.shape
cols = list(range(seqs.shape[1]))
seq_df_yibra = pd.DataFrame(seqs, index=identifiers, columns=cols)
seq_df_yibra.insert(0, 'Library', 'library')
seq_df_yibra.insert(1, 'BC', 'bc')
seq_df_yibra.insert(2, 'ID', 'id')
seq_df_yibra.insert(3, 'Host', 'host')
seq_df_yibra.insert(4, 'Class', 'class')
seq_df_yibra.insert(5, 'Dataset', 'dataset')

# Parse fasta files to link sample metadata to DNA sequence
pattern_lib = 'library\d'
regex_lib = re.compile(pattern_lib)

pattern_bc = 'BC\d\d'
regex_bc = re.compile(pattern_bc)


for index, sample in seq_df_yibra.iterrows():
    library = "empty"
    bc = "empty"
    if regex_lib.search(str(index)):
        library = regex_lib.search(str(index)).group()
    if regex_bc.search(str(index)):
        bc = regex_bc.search(str(index)).group()

    seq_df_yibra.loc[index, 'Library'] = library
    seq_df_yibra.loc[index, 'BC'] = bc


# Now go through metadata excel spreadsheet (in dataframe format)
pattern_nb = '(NB)(\d\d)'
regex_nb = re.compile(pattern_nb)
for index, sample in metadata.iterrows():
    # get the "library" info
    library = sample['YiBRA2_Library_Number']
    sample_NB = sample['YiBRA2_Barcode_Number']

    NB_number = regex_nb.search(sample_NB).group(2)
    barcode = 'BC'+NB_number

    seq_df_yibra.loc[(seq_df_yibra['Library'] == library) & (seq_df_yibra['BC'] == barcode), "ID"] = sample.name
    seq_df_yibra.loc[(seq_df_yibra['Library'] == library) & (seq_df_yibra['BC'] == barcode), "Host"] = sample['Host']

# Select only those that have IDs
seq_df_yibra = seq_df_yibra.loc[seq_df_yibra['ID'] != 'id', :]

# set the label, between fatal and non fatal
seq_df_yibra.loc[:, 'Class'] = 0
seq_df_yibra.loc[seq_df_yibra['Host'] == 'Human Serious or Fatal', 'Class'] = 1
seq_df_yibra.loc[:, 'Dataset'] = 'Yibra'
# seq_df_yibra_original = seq_df_yibra.copy()
# seq_df_yibra_original.shape
#
# seq_df_yibra = seq_df_yibra_original.copy()

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
filename = "../DATA/Marielton.fasta"
# create sequence DataFrame
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(filename, "fasta")]
seqs = np.array([list(str(seq_rec.seq)) for seq_rec in SeqIO.parse(filename, "fasta")])

seqs.shape
cols = list(range(seqs.shape[1]))
seq_df_mari = pd.DataFrame(seqs, index=identifiers, columns=cols)
seq_df_mari.insert(0, 'Library', 'library')
seq_df_mari.insert(1, 'BC', 'bc')
seq_df_mari.insert(2, 'ID', 'id')
seq_df_mari.insert(3, 'Host', 'host')
seq_df_mari.insert(4, 'Class', 'class')
seq_df_mari.insert(5, 'Dataset', 'dataset')

seq_df_mari.loc[:, 'Class'] = 1
seq_df_mari.loc[:, 'Host'] = 'Human'
seq_df_mari['ID'] = seq_df_mari.index
seq_df_mari.loc[:, 'Dataset'] = 'Marielton'


"""
#######################################################################
Talita Cura Dataset
#######################################################################
"""

# Open the fasta file
filename = "../DATA/Talita_cura.fasta"
# create sequence DataFrame
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(filename, "fasta")]
seqs = np.array([list(str(seq_rec.seq)) for seq_rec in SeqIO.parse(filename, "fasta")])

seqs.shape
cols = list(range(seqs.shape[1]))
seq_df_tcura = pd.DataFrame(seqs, index=identifiers, columns=cols)
seq_df_tcura.insert(0, 'Library', 'library')
seq_df_tcura.insert(1, 'BC', 'bc')
seq_df_tcura.insert(2, 'ID', 'id')
seq_df_tcura.insert(3, 'Host', 'host')
seq_df_tcura.insert(4, 'Class', 'class')
seq_df_tcura.insert(5, 'Dataset', 'dataset')

seq_df_tcura.loc[:, 'Class'] = 0
seq_df_tcura.loc[:, 'Host'] = 'Human'
seq_df_tcura['ID'] = seq_df_tcura.index
seq_df_tcura.loc[:, 'Dataset'] = 'T_cura'


"""
#######################################################################
Talita Obitos Dataset
#######################################################################
"""

# Open the fasta file
filename = "../DATA/Talita_obitos.fasta"
# create sequence DataFrame
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(filename, "fasta")]
seqs = np.array([list(str(seq_rec.seq)) for seq_rec in SeqIO.parse(filename, "fasta")])

seqs.shape
cols = list(range(seqs.shape[1]))
seq_df_tob = pd.DataFrame(seqs, index=identifiers, columns=cols)
seq_df_tob.insert(0, 'Library', 'library')
seq_df_tob.insert(1, 'BC', 'bc')
seq_df_tob.insert(2, 'ID', 'id')
seq_df_tob.insert(3, 'Host', 'host')
seq_df_tob.insert(4, 'Class', 'class')
seq_df_tob.insert(5, 'Dataset', 'dataset')

seq_df_tob.loc[:, 'Class'] = 1
seq_df_tob.loc[:, 'Host'] = 'Human'
seq_df_tob['ID'] = seq_df_tob.index
seq_df_tob.loc[:, 'Dataset'] = 'T_obitos'


"""
Data Cleaning
"""

dataframes = [seq_df_yibra, seq_df_mari, seq_df_tcura, seq_df_tob]
seq_df = pd.concat(dataframes)

print("We initially have {0} samples with {1} nucleotides".format(seq_df.shape[0], seq_df.shape[1]))



"""
Data Cleaning
"""


seq_df.replace('N', np.nan, inplace=True)
seq_df.replace('-', np.nan, inplace=True)

# Remove all sequences that contain more than 10% gaps or "N".
threshold = int(seq_df.shape[1]*0.9)
seq_df.dropna(axis=0, how='any', thresh=threshold, inplace=True)

print("After removing samples with more than 10% empty positions, we have {0} samples with {1} nucleotides".format(seq_df.shape[0], seq_df.shape[1]))

# Removes all columns (positions) that contain any gap or "N"
seq_df.dropna(axis=1, how='any', inplace=True)

print("After dropping ALL columns with ANY empty positions, we have {0} samples with {1} nucleotides".format(seq_df.shape[0], seq_df.shape[1]))

# Removes all columns that do not contain variation in the nucleotide, i.e., that are the same for all sequences.
for col in seq_df.columns:
    if seq_df[col].nunique() == 1:
        seq_df.drop(col, axis=1, inplace=True)

print("After removing columns in which there is no variation, we have:\n{0} samples X {1} nucleotides".format(seq_df.shape[0], seq_df.shape[1]))



# How many sequences of each group are we left with?
class0 = seq_df.groupby('Class').count().iloc[0,0]
class1 = seq_df.groupby('Class').count().iloc[1,0]
print("We are left with:\n\t{0} non-severe\n\t{1} severe/fatal".format(class0, class1))


seq_ohe_df = one_hot_encoding(seq_df)

# seq_df_original.to_pickle('../DATA/Human_Analisys/DATA/human_YFV_original_seq_df.pkl')
seq_ohe_df.to_pickle('../DATA/Human_Analisys/DATA/human_YFV_seq_ohe_df.pkl')
seq_df.to_pickle('../DATA/Human_Analisys/DATA/human_YFV_seq_df.pkl')
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
