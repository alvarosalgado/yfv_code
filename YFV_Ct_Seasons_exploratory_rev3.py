#!/usr/bin/env python3
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Alphabet import IUPAC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.ensemble import RandomForestClassifier

import shap

"""
######################################################################
######################################################################
"""
def read_metadata(excel_file):
    metadata_df = pd.read_excel(excel_file, index_col='index')
    metadata_df = metadata_df[['Host', 'Date', 'Ct', 'Virus']]
    return metadata_df


def read_files(file_list, lat_lon_file):
    seq_list = []
    metadata_list = []

    for file in file_list:
        seq_file = file + '.aln'
        metadata_file = file + '.xlsx'

        seq_list.append(seq_file)
        metadata_list.append(metadata_file)

    metadata_dict = {}
    for file in metadata_list:
        metadata_df = read_metadata(file)
        metadata_dict[file] = metadata_df

    metadata = list(metadata_dict.values())
    metadata = pd.concat(metadata)

    index = metadata.index
    index = index.astype('str')
    metadata.index = index

    # Get geographic data (Latitude and Longitude)
    lat_lon = pd.read_csv(lat_lon_file, header=None, names=["ID", "Location", "LAT", "LON"])

    index = pd.Index(lat_lon["ID"])
    index = index.astype('str')
    lat_lon.index = index
    lat_lon.drop("ID", axis=1, inplace=True)

    metadata = metadata.merge(lat_lon, how='inner', left_index=True, right_index=True)

    return metadata

def data_curation(metadata):
    # Remove rows containing NaN or empty values in the Ct column
    metadata = metadata.loc[metadata['Ct'].notnull(), :].copy()

    """
    df[df['A'] > 2]['B'] = new_val  # new_val not set in df
    The warning offers a suggestion to rewrite as follows:

    df.loc[df['A'] > 2, 'B'] = new_val
    """

    # Make sure values in Ct column are float numeric
    metadata['Ct'] = pd.to_numeric(metadata['Ct'])
    metadata['Ct'] = metadata['Ct'].astype(np.float16)

    # Make sure values in Date are datetime
    metadata['Date'] = pd.to_datetime(metadata['Date'])

    # Correct some values
    metadata.replace('Allouata', 'Alouatta', inplace=True)
    metadata.replace('cebidae', 'Cebidae', inplace=True)
    metadata.replace('NHP (unk)', 'unk', inplace=True)
    metadata.replace('Sem informação','unk', inplace=True)
    metadata.replace('Leontopithecus rosalia','L. rosalia', inplace=True)
    metadata['Virus'] = "YFV"

    # Insert another column on the dataset to hold the epidemiologic season
    # 2016/2017
    # 2017/2018
    metadata.insert(2, 'Season', 'season')

    # Fill season values based on date condition:
    # season 1: before August 2017
    # season 2: after August 2017
    mask = metadata['Date'] < pd.datetime(2017,8,1)
    metadata.loc[mask, 'Season'] = '2016/2017'

    mask = metadata['Date'] >= pd.datetime(2017,8,1)
    metadata.loc[mask, 'Season'] = '2017/2018'

    # Insert another column on the dataset to hold the Ct group
    # high = 1
    # low = 0
    metadata.insert(4, 'Ct_Group', 0)
    Ct_threshold = 20
    # Fill Ct groups based on:
    # high: Ct > 20, class 1
    # low: Ct <= 20, class 0
    mask = metadata['Ct'] <= Ct_threshold
    metadata.loc[mask, 'Ct_Group'] = 0

    mask = metadata['Ct'] > Ct_threshold
    metadata.loc[mask, 'Ct_Group'] = 1


    return metadata

def plot_scatter(host_list, metadata):
    n_host = len(host_list)
    fig1, axes1 = plt.subplots(figsize=(6, 24), nrows=n_host, ncols=1, sharex=True)

    i = 0

    for host in host_list:
        df1 = metadata[metadata["Host"] == host]
        x1 = df1['Date'].values
        y1 = df1['Ct'].values

        axes1[i].scatter(x1, y1)
        axes1[i].set_title(host + ' Ct values')

        axes1[i].set_xlabel('Date')
        axes1[i].set_ylabel('Ct')

        axes1[i].set_ylim((0, 40))

        i += 1

    fig1.autofmt_xdate()

    fig1.savefig('./figures/Scatter_seasons_geo.png', format='png', dpi=300, transparent=False)

def plot_boxplot():

    fig2, axes2 = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, sharey=True)
    data1_17 = df1[df1['Date'] <= pd.datetime(2017,8,1)]
    data1_18 = df1[df1['Date'] > pd.datetime(2017,8,1)]
    data2_17 = df2[df2['Date'] <= pd.datetime(2017,8,1)]
    data2_18 = df2[df2['Date'] > pd.datetime(2017,8,1)]

    axes2[0].boxplot([data1_17['Ct'], data1_18['Ct']], labels=['2016/2017', '2017/2018'])
    axes2[0].set_title(host1 + ' Ct values boxplot per season')
    axes2[0].set_xlabel('Season')
    axes2[0].set_ylabel('Ct')
"""
######################################################################
MAIN
######################################################################
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

lat_lon_file = "lat_long.txt"

metadata = read_files(file_list, lat_lon_file)
metadata = data_curation(metadata)

"""
The SettingWithCopyWarning was created to flag potentially confusing "chained" assignments, such as the following, which don't always work as expected, particularly when the first selection returns a copy. [see GH5390 and GH5597 for background discussion.]

df[df['A'] > 2]['B'] = new_val  # new_val not set in df
The warning offers a suggestion to rewrite as follows:

df.loc[df['A'] > 2, 'B'] = new_val
However, this doesn't fit your usage, which is equivalent to:

df = df[df['A'] > 2]
df['B'] = new_val
While it's clear that you don't care about writes making it back to the original frame (since you overwrote the reference to it), unfortunately this pattern can not be differentiated from the first chained assignment example, hence the (false positive) warning. The potential for false positives is addressed in the docs on indexing, if you'd like to read further. You can safely disable this new warning with the following assignment.

pd.options.mode.chained_assignment = None  # default='warn'
"""

metadata.to_csv("../Callithrix_Analysis/DATA/!CLEAN/metadata_YFV_NHP_GEO.csv")

host_list = list(metadata['Host'].unique())
plot_scatter(host_list, metadata)

metadata_low = metadata[metadata["Ct"] < 20].copy()

low_ct_mean = metadata_low.groupby('Host')["Ct"].mean()

host_count = metadata.groupby('Host')['Host'].count()
