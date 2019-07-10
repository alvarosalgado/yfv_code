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


# Read excel into a pd.DataFrame
file = '../DATA/2018-01_Salvador/CONSENSUS/YFV_Jan_2018_SampleList.xlsx'
sample_list_excel = pd.read_excel(file, index_col='YiBRA_SSA_ID')

# Select only rows containing human samples
human_df1 = sample_list_excel[sample_list_excel['Host']=='Human']
human_df2 = sample_list_excel[sample_list_excel['Host']=='Human Grave']
human_df = pd.concat([human_df1, human_df2], axis=0)

# Remove samples that were not sequenced
human_df = human_df[pd.notnull(human_df['YiBRA2_Library_Number'])]

"""####################################"""


# check if there are only YFV samples
# human_df['Original_Lab_Results']

human_df.loc[human_df['YiBRA2_Library_Number'] == 'library 4', 'YiBRA2_Library_Number'] = 'library4'

human_df.loc[human_df['YiBRA2_Library_Number'] == 'library 5', 'YiBRA2_Library_Number'] = 'library5'

human_df.loc[human_df['YiBRA2_Library_Number'] == 'library 6', 'YiBRA2_Library_Number'] = 'library6'

human_df.loc[human_df['YiBRA2_Library_Number'] == 'library 7', 'YiBRA2_Library_Number'] = 'library7'

"""####################################"""


file_list = glob.glob("../DATA/2018-01_Salvador/CONSENSUS/*.fasta")

"""####################################"""

pattern1 = 'library\d'
regex1 = re.compile(pattern1)

seq_dict = {}
for filename in file_list:
    # Search library id
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

seq_dict['library2'].head(12)

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
seq_df.loc[seq_df['Host'] == 'Human Grave', 'Class'] = 1
seq_df.replace('N', np.nan, inplace=True)
seq_df.replace('-', np.nan, inplace=True)

threshold = int(seq_df.shape[1]*0.9)
seq_df.dropna(axis=0, how='any', thresh=threshold, inplace=True)
seq_df.shape

seq_df.dropna(axis=1, how='any', inplace=True)

seq_df.groupby('Host').count()
# we are left with 17 normal human cases and 4 serious human cases
# check if these 4 cases are deaths or just serious.
