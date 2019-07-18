#!/usr/bin/env python3
"""
#######################################################################

- Specific for "YFV Ct Callithrix investigation"

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
import seaborn as sns; sns.set_style('whitegrid')

import re

"""
#######################################################################

#######################################################################
"""

def read_files(file_list):
    seq_list = []
    metadata_list = []

    for file in file_list:
        seq_file = file + '.aln'
        metadata_file = file + '.xlsx'

        seq_list.append(seq_file)
        metadata_list.append(metadata_file)

    return (seq_list, metadata_list)

def create_seq_df(file, seq_start=0):
    # Creates a dataframe based on a ".aln" file.

    # Gets the sequences IDs from a multi-fasta into a list
    identifiers = [seq_rec.id for seq_rec in SeqIO.parse(file, "clustal")]

    # Gets the sequences nucleotides, for each sequence in a multi-fasta
    seqs = np.array([list(str(seq_rec.seq)) for seq_rec in SeqIO.parse(file, "clustal")])

    # Creates columns names based on position, starting from 0.
    cols = list(range(seq_start, seq_start + seqs.shape[1]))

    # Creates dataframe with data
    seq_df = pd.DataFrame(seqs, index=identifiers, columns=cols)

    return seq_df


def create_seq_dict(seq_list):
    seq_dict = {}
    for file in seq_list:
        df = create_seq_df(file)
        seq_dict[file] = df
    return seq_dict


def create_meta_dict(metadata_list):
    metadata_dict = {}
    for file in metadata_list:
        metadata_df = pd.read_excel(file, index_col='index')
        metadata_dict[file] = metadata_df
    return metadata_dict

def link_meta_info(file_list, seq_dict, metadata_dict):
    index_bookeeping = {} # to avoid matching multiple sequences for the same ID.
    index_search = {}     #Just an auxiliary variable I used to see if all indexes were being counted.
    count = 0

    for file in file_list: # compare seq_df to metadata_df in a pairwise manner.
        seq_file = file + '.aln'
        metadata_file = file + '.xlsx'

        # Here I hold both the sequence df and the metadata df, to merge information.
        seq_df = seq_dict[seq_file]
        metadata_df = metadata_dict[metadata_file]

        # Prepare seq_df to receive the metadata info.
        seq_df.insert(0, 'ID', 'id')
        seq_df.insert(1, 'Host', 'host')
        seq_df.insert(2, 'Ct', 'ct')
        seq_df.insert(3, 'Date', 'date')


        # For each ID in metadata (here in its index).
        # In the excel files, there is a column called "index".
        # This column was used as the "metadata dataframe" index.
        # So I iterate over these indexes and look for them (try to match them using regex) in the fasta file ID.

        # for each index, and each metadata related to that index.
        for index_meta, meta in metadata_df.iterrows():

            #Just an auxiliary variable I used to see if all indexes were being counted.
            if index_meta not in index_search:
                index_search[index_meta] = 1
            else:
                index_search[index_meta] += 1

            # I have pre-edited the fasta files ID fields to put the index values between vertical bars "|".
            # This was to make it easier to create a pattern and use regex.
            pattern = '\|' + str(index_meta) + '\|'
            regex = re.compile(pattern)

            # For each fasta ID (index_seq) in the file...
            for index_b, sample_b in seq_df.iterrows():
                # If the metadata index is in this fasta ID...
                if regex.search(index_b):
                    # if this sequence still has no metadata values associated...
                    if seq_df.loc[index_b,'ID'] == 'id':
                        # fill in metadata values to seq_df
                        seq_df.loc[index_b,'ID'] = index_meta
                        seq_df.loc[index_b,'Host'] = metadata_df.loc[index_meta, 'Host']
                        seq_df.loc[index_b,'Date'] = metadata_df.loc[index_meta, 'Date']
                        seq_df.loc[index_b,'Ct'] = metadata_df.loc[index_meta, 'Ct']

                        index_bookeeping[index_meta] = 1
                    # else, if this sequence already has metadata values associated
                    # (this happens because identical sequences are grouped together, and their fasta IDs
                    # keep all the information of all these sequences)
                    # and if this is the first time this specific index_meta is matched...
                    elif index_meta not in index_bookeeping:
                        # Copy the sequence, but with new metadata, and append it to seq_df
                        sample_copy = pd.Series(sample_b)
                        index_copy = str(index_meta+'_')+str(index_b)
                        sample_copy.name = index_copy
                        seq_df.append(sample_copy)
                        seq_df.loc[index_copy,'ID'] = index_meta
                        seq_df.loc[index_copy,'Host'] = metadata_df.loc[index_meta, 'Host']
                        seq_df.loc[index_copy,'Date'] = metadata_df.loc[index_meta, 'Date']
                        seq_df.loc[index_copy,'Ct'] = metadata_df.loc[index_meta, 'Ct']

                        index_bookeeping[index_meta] = 1
                    else:
                        index_bookeeping[index_meta] += 1


def concat_seq_df(seq_dict):

    dfs = list(seq_dict.values())
    seq_df = pd.concat(dfs)

    return seq_df

"""
Data Cleaning
"""
def clean_df(seq_df, threshold=0.9):

    # First, turn all "N" and "-" into np.nan. This will mark the missing values.
    seq_df.replace('N', np.nan, inplace=True)
    seq_df.replace('-', np.nan, inplace=True)

    # Second, keep only rows (samples) containing less then threshold% missing values (NaN).
    threshold = int(seq_df.shape[1]*threshold)
    seq_df.dropna(axis=0, how='any', thresh=threshold, inplace=True)

    # Third, remove all columns that still containg missing values.
    seq_df.dropna(axis=1, how='any', inplace=True)

    # Remove rows containing NaN or empty values in the Ct column
    seq_df = seq_df[seq_df['Ct'].notnull()]
    seq_df = seq_df[seq_df['Ct'] != 'ct']

    # Make sure values in Ct column are float numeric
    seq_df['Ct'] = pd.to_numeric(seq_df['Ct'])
    seq_df['Ct'] = seq_df['Ct'].astype(np.float16)

    # Make sure values in Date are datetime
    seq_df['Date'] = pd.to_datetime(seq_df['Date'])

    # Correct some values
    seq_df.replace('Allouata', 'Alouatta', inplace=True)
    seq_df.replace('cebidae', 'Cebidae', inplace=True)
    seq_df.replace('NHP (unk)', 'unk', inplace=True)
    seq_df.replace('Sem informação','unk', inplace=True)
    seq_df.replace('Leontopithecus rosalia','L. rosalia', inplace=True)

    # Remove all columns that contain only one value (they have no information for the classification algorithm)
    for col in seq_df.columns:
        if seq_df[col].nunique() == 1:
            seq_df.drop(col, axis=1, inplace=True)

    return seq_df


def insert_features(seq_df, Ct_threshold = 20):
    # Insert another column on the dataset to hold the epidemiologic season
    # 2016/2017
    # 2017/2018
    seq_df.insert(4, 'Season', 'season')

    # Fill season values based on date condition:
    # season 1: before August 2017
    # season 2: after August 2017
    mask = seq_df['Date'] < pd.datetime(2017,8,1)
    seq_df.loc[mask, 'Season'] = '2016/2017'

    mask = seq_df['Date'] >= pd.datetime(2017,8,1)
    seq_df.loc[mask, 'Season'] = '2017/2018'

    # Insert another column on the dataset to hold the Ct group
    # high = 1
    # low = 0
    seq_df.insert(5, 'Ct_Group', 0)
    # Fill Ct groups based on:
    # high: Ct > 20
    # low: Ct <= 20
    mask = seq_df['Ct'] <= Ct_threshold
    seq_df.loc[mask, 'Ct_Group'] = 0

    mask = seq_df['Ct'] > Ct_threshold
    seq_df.loc[mask, 'Ct_Group'] = 1

    return seq_df


def one_hot_encoding(seq_df, file_path='../Callithrix_Analysis/DATA/!CLEAN/'):

    nucleotides_df = seq_df.iloc[:, 6:]
    seq_ohe_df = pd.get_dummies(nucleotides_df)

    seq_ohe_df.insert(0, 'ID', seq_df['ID'])
    seq_ohe_df.insert(1, 'Host', seq_df['Host'])
    seq_ohe_df.insert(2, 'Ct', seq_df['Ct'])
    seq_ohe_df.insert(3, 'Date', seq_df['Date'])
    seq_ohe_df.insert(4, 'Season', seq_df['Season'])
    seq_ohe_df.insert(5, 'Ct_Group', seq_df['Ct_Group'])


    seq_ohe_df.to_csv(file_path + 'YFV_seq_ohe_df.csv', index=True, header=True, decimal='.', sep=',', float_format='%.2f')

    seq_ohe_df.to_pickle(file_path + 'YFV_seq_ohe_df.pkl')

    seq_df.to_pickle(file_path + 'YFV_seq_df.pkl')

    return seq_ohe_df

"""
#######################################################################
MAIN
#######################################################################
"""


file_1 = '../Callithrix_Analysis/DATA/!CLEAN/2019-01-30_ZIBRA2_YFV-RIO-Diferentes_CTs'
file_2 = '../Callithrix_Analysis/DATA/!CLEAN/NHP_65_outbreak'
file_3 = '../Callithrix_Analysis/DATA/!CLEAN/2018-01_Salvador'
file_4 = '../Callithrix_Analysis/DATA/!CLEAN/2018-03-04_LACEN_Bahia'
file_5 = '../Callithrix_Analysis/DATA/!CLEAN/FUNED_AGOSTO-2018'
file_6 = '../Callithrix_Analysis/DATA/!CLEAN/RIO_DE_JANEIRO'
file_7 = '../Callithrix_Analysis/DATA/!CLEAN/YFV_LACEN_BAHIA'

file_list = [file_1,
            file_2,
            file_3,
            file_4,
            file_5,
            file_6,
            file_7]

(seq_list, metadata_list) = read_files(file_list)

seq_dict = create_seq_dict(seq_list)
metadata_dict = create_meta_dict(metadata_list)

link_meta_info(file_list, seq_dict, metadata_dict)
# seq_dict[file_5+'.aln']
seq_df = concat_seq_df(seq_dict)

# seq_df.loc['FUNED_AGOSTO-2018|FAH50662/|BC33|_FAH50662.primertrimmed.sorted.bam', :]
# metadata_dict[file_7+'.xlsx'].head()
seq_df = clean_df(seq_df, threshold=0.9)
seq_df= insert_features(seq_df, Ct_threshold = 20)

# seq_0 = seq_df.iloc[0, :].values[6:]
# seq_df.replace('N', np.nan, inplace=True)
# seq_df.replace('-', np.nan, inplace=True)
# threshold = int(seq_df.shape[1]*0.9)
# seq_df.dropna(axis=0, how='any', thresh=threshold, inplace=True)
# seq_df.dropna(axis=1, how='any', inplace=True)
# seq_df.shape

seq_ohe_df = one_hot_encoding(seq_df, file_path='../Callithrix_Analysis/DATA/!CLEAN/')

host_count = seq_df.groupby('Host')["ID"].count()
host_count = host_count[["Alouatta", "Callithrix"]]
host_count.name = "Number of Sequences"
table_count1_latex = host_count.to_latex()
with open('./tables/table_count1_latex.txt', 'w') as f:
    f.write(table_count1_latex)

listdf = []
for host, host_data in seq_df.groupby('Host'):
    if ((host == "Alouatta") or (host == "Callithrix")):
        df = pd.DataFrame(host_data.groupby('Ct_Group')['ID'].count()).T
        df.index = [host]
        df.columns = ["Low Ct", "High Ct"]
        listdf.append(df)
    #df.index = [host]
    #print(b)
df = pd.concat(listdf)
table_count2_latex = df.to_latex()
with open('./tables/table_count2_latex.txt', 'w') as f:
    f.write(table_count2_latex)
