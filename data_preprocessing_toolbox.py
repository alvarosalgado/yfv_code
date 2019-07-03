#!/usr/bin/env python3

from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Alphabet import IUPAC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import re

"""
#######################################################################
- Read the fasta files containing the sequences' DNAs and the excel files containing their metadata.
- Clean the data.
- Turn it into "one-hot encoded" data.
- Save it to csv and pkl.
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

def create_seq_df(file):
    # Creates a dataframe based on a ".aln" file.

    # Gets the sequences IDs from a multi-fasta into a list
    identifiers = [seq_rec.id for seq_rec in SeqIO.parse(file, "clustal")]

    # Gets the sequences nucleotides, for each sequence in a multi-fasta
    seqs = np.array([list(str(seq_rec.seq)) for seq_rec in SeqIO.parse(file, "clustal")])

    # Creates columns names based on position, starting from 0.
    cols = list(range(seqs.shape[1]))

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
