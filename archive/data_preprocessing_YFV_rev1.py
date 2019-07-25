# %%
%matplotlib inline

from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Alphabet import IUPAC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.ensemble import RandomForestClassifier

import shap
# %%
shap.__version__
# %% markdown
# # Readme
# With this notebook:
# - Read the fasta files containing the sequences' DNAs and the excel files containing their metadata.
# - Clean the data.
# - Turn it into "one-hot encoded" data.
# - Save it to csv and pkl.
# %% markdown
# ### NHP Sequence alignment files (.aln format)
# %% markdown
# Put the files names (relative paths) in a list.
#
# We will iterate over this list to read the sequences into dataframes.
# %%
file_1 = '../DATA/!CLEAN/2019-01-30_ZIBRA2_YFV-RIO-Diferentes_CTs'

file_2 = '../DATA/!CLEAN/NHP_65_outbreak'

file_3 = '../DATA/!CLEAN/2018-01_Salvador'

file_4 = '../DATA/!CLEAN/2018-03-04_LACEN_Bahia'

file_5 = '../DATA/!CLEAN/FUNED_AGOSTO-2018'

file_6 = '../DATA/!CLEAN/RIO_DE_JANEIRO'

file_7 = '../DATA/!CLEAN/YFV_LACEN_BAHIA'
# %%
file_list = [file_1,
            file_2,
            file_3,
            file_4,
            file_5,
            file_6,
            file_7]
seq_list = []
metadata_list = []
# %%
for file in file_list:
    seq_file = file + '.aln'
    metadata_file = file + '.xlsx'

    seq_list.append(seq_file)
    metadata_list.append(metadata_file)
# %% markdown
# ### Dataframes containing NHP YFV sequences
# A function to iterate over all file names and create a dataframe containing the **nucleotide sequences** for each one, putting them in a _dictionary_.
# %%
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
# %%
seq_dict = {}
for file in seq_list:
    print(file)
    df = create_seq_df(file)
    seq_dict[file] = df
# %%
sizes = [len(seq_dict[file]) for file in seq_list]
n_seqs = sum(sizes)
print(sizes)
print(n_seqs)
# %% markdown
# ### NHP Metadata
# The following code reads the excel spreadsheet containing the metadata related to the sequences and includes them in the dataframe.
#
# One spreadsheet per group of sequences from ZIBRA database, all into a _dictionary_.
#
# I only keep the information I'm going to use now, i.e., 'Host', 'Date' and 'Ct'.
#
# I use regex to link the spreadsheet to the sequences.

# %%
def read_metadata(excel_file):
    metadata_df = pd.read_excel(excel_file, index_col='index')
    metadata_df = metadata_df[['Host', 'Date', 'Ct']]
    return metadata_df
# %%

# %%
metadata_dict = {}
for file in metadata_list:
    print(file)
    metadata_df = read_metadata(file)
    metadata_dict[file] = metadata_df
# %%
sizes = [len(metadata_dict[file]) for file in metadata_list]
n_meta = sum(sizes)
print(sizes)
print(n_meta)
# %%
indexes = [metadata_dict[file].index for file in metadata_list]
sum([len(index) for index in indexes])
# %% markdown
# # Regex - Merge information on metadata to dna sequence dataframe
# Parse through metadata and sequences IDs, linking information and adding it to the seqs dataframes.
# %%
import re
# %%
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

# %%
#index_search
# %%
#len(index_search)
# %% markdown
# Both the `index_search` and the `len(index_search)` results show that there are duplicate metadata in the spreadsheets. I manually checked it, and it is indeed duplicated, with the exact values (it is consistent).
#
# So there is no problem here.
#
# The difference between the number of metadata and sequences is due to identical sequences and some low quality sequences I manually removed.
# %% markdown
# ### Merge all dataframes into one
# %%
dfs = list(seq_dict.values())
# %%
len(dfs)
# %%
for df in dfs:
    print(df.shape)
# %%
seq_df = pd.concat(dfs)
# %%
seq_df
# %% markdown
# # Data Cleaning
# %% markdown
# ## Clear missing values
#
# - First, turn all "N" and "-" into `np.nan`. This will mark the missing values.
# %%
seq_df.replace('N', np.nan, inplace=True)
seq_df.replace('-', np.nan, inplace=True)
# %% markdown
# - Second, keep only rows (samples) containing less then 5% missing values (NaN).
# %%
threshold = int(seq_df.shape[1]*0.9)
# %%
seq_df.dropna(axis=0, how='any', thresh=threshold, inplace=True)
# %%
seq_df.shape
# %% markdown
# - Third, remove all columns that still containg missing values.
# %%
seq_df.dropna(axis=1, how='any', inplace=True)
# %%
seq_df.shape
# %%
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

# Show all hosts present
print(seq_df['Host'].unique())
# %% markdown
# ## Remove all columns that contain only one value (they have no information for the classification algorithm)
# %%
for col in seq_df.columns:
    #print(col)
    if seq_df[col].nunique() == 1:
        seq_df.drop(col, axis=1, inplace=True)


# %%
seq_df.shape
# %%

# %%

# %%
seq_df.head()
# %% markdown
# ## Seasons
# Prepare data to perform "epidemic season" analysis
# %%
# Insert another column on the dataset to hold the epidemiologic season
# 2016/2017
# 2017/2018
seq_df.insert(4, 'Season', 'season')
# %%
# Fill season values based on date condition:
# season 1: before August 2017
# season 2: after August 2017
mask = seq_df['Date'] < pd.datetime(2017,8,1)
seq_df.loc[mask, 'Season'] = '2016/2017'

mask = seq_df['Date'] >= pd.datetime(2017,8,1)
seq_df.loc[mask, 'Season'] = '2017/2018'

seq_df.head()
# %% markdown
# ## High/Low Ct
# Prepare data to perform "high/low Ct" analysis
# %%
# Insert another column on the dataset to hold the Ct group
# high = 1
# low = 0
seq_df.insert(5, 'Ct_Group', 0)
# %%
Ct_threshold = 20
# %%
# Fill Ct groups based on:
# high: Ct > 20
# low: Ct <= 20
mask = seq_df['Ct'] <= Ct_threshold
seq_df.loc[mask, 'Ct_Group'] = 0

mask = seq_df['Ct'] > Ct_threshold
seq_df.loc[mask, 'Ct_Group'] = 1

seq_df.head()
# %%
seq_df.groupby('Host')["ID"].count()
# %% markdown
# ### We are left with a dataset containing 26 Alouatta samples and 27 Callithrix samples.
# - of the Callithrix samples, 6 are high Ct $(> 20)$ and 21 are low Ct $(< 20)$
# %%
for host, host_data in seq_df.groupby('Host'):
    print(host)
    print(host_data.groupby('Ct_Group')['ID'].count(), '\n')
    #print(b)
# %%

# %% markdown
# # One hot encoding
# %%
nucleotides_df = seq_df.iloc[:, 6:]
# %%
nucleotides_df.head()
# %%
seq_ohe_df = pd.get_dummies(nucleotides_df)
seq_ohe_df.head()
# %%
seq_ohe_df.shape
# %%
seq_df.shape
# %%
seq_ohe_df.index == seq_df.index
# %%

# %%

# %%
seq_ohe_df.insert(0, 'ID', seq_df['ID'])
seq_ohe_df.insert(1, 'Host', seq_df['Host'])
seq_ohe_df.insert(2, 'Ct', seq_df['Ct'])
seq_ohe_df.insert(3, 'Date', seq_df['Date'])
seq_ohe_df.insert(4, 'Season', seq_df['Season'])
seq_ohe_df.insert(5, 'Ct_Group', seq_df['Ct_Group'])
# %%
seq_ohe_df.head()
# %% markdown
# # Save to .csv and .pkl
# %%
seq_ohe_df.to_csv('../DATA/!CLEAN/YFV_seq_ohe_df.csv', index=True, header=True, decimal='.', sep=',', float_format='%.2f')
# %%
seq_ohe_df.to_pickle('../DATA/!CLEAN/YFV_seq_ohe_df.pkl')
# %%
seq_df.to_pickle('../DATA/!CLEAN/YFV_seq_df.pkl')
# %%
