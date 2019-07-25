#!/usr/bin/env python3

# %%

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
# ### NHP Sequence alignment files (.aln format)
# %% markdown
# Put the files names (relative paths) in a list.
#
# We will iterate over this list to read the sequences into dataframes.
# %%
file_1 = '../Callithrix_Analysis/DATA/!CLEAN/2019-01-30_ZIBRA2_YFV-RIO-Diferentes_CTs'

file_2 = '../Callithrix_Analysis/DATA/!CLEAN/NHP_65_outbreak'

file_3 = '../Callithrix_Analysis/DATA/!CLEAN/2018-01_Salvador'

file_4 = '../Callithrix_Analysis/DATA/!CLEAN/2018-03-04_LACEN_Bahia'

file_5 = '../Callithrix_Analysis/DATA/!CLEAN/FUNED_AGOSTO-2018'

file_6 = '../Callithrix_Analysis/DATA/!CLEAN/RIO_DE_JANEIRO'

file_7 = '../Callithrix_Analysis/DATA/!CLEAN/YFV_LACEN_BAHIA'
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
# ### NHP Metadata
# The following code reads the excel spreadsheet containing the metadata related to the sequences and includes them in the dataframe.
#
# One spreadsheet per group of sequences from ZIBRA database, all into a _dictionary_.
#
# I only keep the information I'm going to use now, i.e., 'Host', 'Date', 'Ct' and 'Virus'.
#
# I use regex to link the spreadsheet to the sequences.

# %%
def read_metadata(excel_file):
    metadata_df = pd.read_excel(excel_file, index_col='index')
    metadata_df = metadata_df[['Host', 'Date', 'Ct', 'Virus']]
    return metadata_df
# %%
metadata_dict = {}
for file in metadata_list:
    # print(file)
    metadata_df = read_metadata(file)
    metadata_dict[file] = metadata_df
# %%
# Checking the siza of the metadata
sizes = [len(metadata_dict[file]) for file in metadata_list]
n_meta = sum(sizes)
print(sizes)
print(n_meta)
# %%
indexes = [metadata_dict[file].index for file in metadata_list]
sum([len(index) for index in indexes])
# %% markdown
# ### Merge all dataframes into one
# %%
metadata = list(metadata_dict.values())
# %%
metadata = pd.concat(metadata)

# Get geographic data (Latitude and Longitude)
lat_lon = pd.read_csv("lat_long.txt", header=None, names=["ID", "Location", "LAT", "LON"])

index = pd.Index(lat_lon["ID"])
index = index.astype('str')
lat_lon.index = index
lat_lon.drop("ID", axis=1, inplace=True)

lat_lon.loc['RJ259', :]

index = metadata.index
index = index.astype('str')
metadata.index = index

metadata.loc['RJ259', :]

lat_lon.shape
metadata.shape

# Join metadata and geo info.
metadata = metadata.merge(lat_lon, how='inner', left_index=True, right_index=True)

metadata.loc['RJ259', :]

for i in range(metadata.shape[0]):
    print(metadata.iloc[i, 2])


metadata[metadata["LAT"]==" NA"]['Ct']
metadata.loc['NTC', :]
# lat_lon[lat_lon["Location"]==" ES Itarana"]
# %%
# metadata.shape
# %% markdown
# ### Data Cleaning
# %%
# Remove rows containing NaN or empty values in the Ct column
metadata = metadata[metadata['Ct'].notnull()]
#metadata = metadata[metadata['Ct'] != 'ct']

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

# Show all hosts present
# print(metadata['Host'].unique())
#print(metadata.head())
# %%
#metadata
# %% markdown
# # Data exploration
#
# The next session explores how Ct value changes by species and by epidemic period.
# The initial hypothesis was that Ct values of Callithrix samples were increasing. However, the data analysis below shows it's not the case.
# %%
# First quick verification.
# Mean Ct values per host type:
mean_cts = metadata.groupby('Host')['Ct'].mean()
fig1, axes1 = plt.subplots(figsize=(10, 6))
axes1.bar(mean_cts.index, mean_cts)
axes1.set_title("Mean Ct values per host - Full Dataset")
axes1.set_ylabel("Ct Value")
for tick in axes1.get_xticklabels():
    tick.set_rotation(45)
fig1.tight_layout()
fig1.savefig("./figures/Mean_Ct_full_dataset.png", format='png', dpi=300, transparent=False)

# %% markdown
# ## Seasons
# %%
# Insert another column on the dataset to hold the epidemiologic season
# 2016/2017
# 2017/2018
metadata.insert(2, 'Season', 'season')
# %%
# Fill season values based on date condition:
# season 1: before August 2017
# season 2: after August 2017
mask = metadata['Date'] < pd.datetime(2017,8,1)
metadata.loc[mask, 'Season'] = '2016/2017'

mask = metadata['Date'] >= pd.datetime(2017,8,1)
metadata.loc[mask, 'Season'] = '2017/2018'

# metadata.head()
# %% markdown
# ## High and Low Ct's
# %%
# Insert another column on the dataset to hold the Ct group
# high = 1
# low = 0
metadata.insert(4, 'Ct_Group', 0)
# %%
Ct_threshold = 20
# %%
# Fill Ct groups based on:
# high: Ct > 20, class 1
# low: Ct <= 20, class 0
mask = metadata['Ct'] <= Ct_threshold
metadata.loc[mask, 'Ct_Group'] = 0

mask = metadata['Ct'] > Ct_threshold
metadata.loc[mask, 'Ct_Group'] = 1

# metadata.head()
# %% markdown
# ## First quick verification
# %%
# First quick verification.
# Mean Ct values per host type:
fig2, axes2 = plt.subplots(figsize=(10, 6), nrows=1, ncols=2, sharey=True)
i=0
for group, group_df in metadata.groupby('Season'):
    mean_cts = group_df.groupby('Host')['Ct'].mean()
    axes2[i].bar(mean_cts.index, mean_cts)
    axes2[i].set_title("Mean Ct values per host for season {0} ({1})\n".format(i+1, group))
    axes2[i].set_ylabel("Ct Value")
    for tick in axes2[i].get_xticklabels():
        tick.set_rotation(45)
    i += 1
fig2.tight_layout()
fig2.savefig("./figures/Mean_Ct_per_host_seasons.png", format='png', dpi=300, transparent=False)

# %%
# df_list = []
# for host, host_data in metadata.groupby('Host'):
#     # print(host)
#     ct = host_data.groupby('Season')['Ct'].mean()
#     df = pd.DataFrame(ct).T
#     df.index = pd.Index([host], name='Host')
#     df_list.append(df)
# table_mean_Ct = pd.concat(df_list)
# table_mean_Ct_latex = table_mean_Ct.to_latex()
# with open('./tables/table_mean_Ct_latex.txt', 'w') as f:
#     f.write(table_mean_Ct_latex)
table_mean_Ct = metadata.groupby('Host')["Ct"].mean()
table_mean_Ct_latex = table_mean_Ct.to_latex()
with open('./tables/table_mean_Ct_latex.txt', 'w') as f:
    f.write(table_mean_Ct_latex)

# %%
import seaborn as sns
sns.set(style="whitegrid")
# %%
meta_cal_alo = metadata[(metadata['Host'] == 'Callithrix') | (metadata['Host'] == 'Alouatta')]
# %%
# Separate two datasets, one for each host

callithrix_df = metadata[metadata['Host'] == 'Callithrix']
alouatta_df = metadata[metadata['Host'] == 'Alouatta']
# %% markdown
# ### Do we have a balanced dataset?
# %%
with open('./text/balanced_data.txt', 'w') as f:
    f.write("Total {0} samples: {1}\n".format(callithrix_df["Host"].iloc[0], len(callithrix_df)))
    f.write("{0} samples in Season {1}: {2}\n".format(callithrix_df["Host"].iloc[0], "2016/2017", len(callithrix_df[callithrix_df["Season"]=="2016/2017"])))
    f.write("{0} samples in Season {1}: {2}\n".format(callithrix_df["Host"].iloc[0], "2017/2018", len(callithrix_df[callithrix_df["Season"]=="2017/2018"])))
    # %%
    f.write("Total {0} samples: {1}\n".format(callithrix_df["Host"].iloc[0], len(callithrix_df)))
    f.write("{0} samples with high Ct: {1}\n".format(callithrix_df["Host"].iloc[0], len(callithrix_df[callithrix_df["Ct_Group"]==1])))
    f.write("{0} samples with low Ct: {1}\n".format(callithrix_df["Host"].iloc[0], len(callithrix_df[callithrix_df["Ct_Group"]==0])))
    # %%
    f.write("Total {0} samples: {1}\n".format(alouatta_df["Host"].iloc[0], len(alouatta_df)))
    f.write("{0} samples in Season {1}: {2}\n".format(alouatta_df["Host"].iloc[0], "2016/2017", len(alouatta_df[alouatta_df["Season"]=="2016/2017"])))
    f.write("{0} samples in Season {1}: {2}\n".format(alouatta_df["Host"].iloc[0], "2017/2018", len(alouatta_df[alouatta_df["Season"]=="2017/2018"])))
    # %%
    f.write("Total {0} samples: {1}\n".format(alouatta_df["Host"].iloc[0], len(alouatta_df)))
    f.write("{0} samples with high Ct: {1}\n".format(alouatta_df["Host"].iloc[0], len(alouatta_df[alouatta_df["Ct_Group"]==1])))
    f.write("{0} samples with low Ct: {1}\n".format(alouatta_df["Host"].iloc[0], len(alouatta_df[alouatta_df["Ct_Group"]==0])))

# %% markdown
# ## Understanding how Ct values changed by season
# The function below plots boxplots, violin plots and scatter plots comparing the Ct value for both seasons.
# %%
def plot_figures(host1, host2, df1, df2):
    fig1, axes1 = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, sharey=True)

    x1 = df1['Date'].values
    y1 = df1['Ct'].values

    x2 = df2['Date'].values
    y2 = df2['Ct'].values

    axes1[0].scatter(x1, y1)
    axes1[0].set_title(host1 + ' Ct values')

    axes1[0].set_xlabel('Sample Collection Date')
    axes1[0].set_ylabel('Ct Value by Sample')

    axes1[1].scatter(x2, y2)
    axes1[1].set_title(host2 + ' Ct values')

    axes1[1].set_xlabel('Sample Collection Date')
    axes1[1].set_ylabel('Ct Value by Sample')

    fig1.autofmt_xdate()

    fig1.savefig('./figures/Scatter_seasons.png', format='png', dpi=300, transparent=False)


    fig2, axes2 = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, sharey=True)
    data1_17 = df1[df1['Date'] <= pd.datetime(2017,8,1)]
    data1_18 = df1[df1['Date'] > pd.datetime(2017,8,1)]
    data2_17 = df2[df2['Date'] <= pd.datetime(2017,8,1)]
    data2_18 = df2[df2['Date'] > pd.datetime(2017,8,1)]

    axes2[0].boxplot([data1_17['Ct'], data1_18['Ct']], labels=['2016/2017', '2017/2018'])
    axes2[0].set_title(host1 + ' Ct values boxplot per season')
    axes2[0].set_xlabel('Season')
    axes2[0].set_ylabel('Ct')

    axes2[1].boxplot([data2_17['Ct'], data2_18['Ct']], labels=['2016/2017', '2017/2018'])
    axes2[1].set_title(host2 + ' Ct values boxplot per season')
    axes2[1].set_xlabel('Season')
    axes2[1].set_ylabel('Ct')

    fig2.savefig('./figures/BoxPlot_seasons.png', format='png', dpi=300, transparent=False)


    fig3, axes3 = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, sharey=True)
    sns.violinplot(x="Season", y="Ct", data=df1, ax=axes3[0])
    sns.violinplot(x="Season", y="Ct", data=df2, ax=axes3[1])
    axes3[0].set_title(host1 + ' Ct values violin plot per season')
    axes3[1].set_title(host2 + ' Ct values violin plot per season');

    fig3.savefig('./figures/ViolinPlot.png', format='png', dpi=300, transparent=False)
# %%

# %%
plot_figures('Callithrix', 'Alouatta', callithrix_df, alouatta_df)
# %% markdown
# ## Understanding how Ct values changed by High/Low groups

# %%
def plot_figures_2(host1, host2, df1, df2):
    fig1, axes1 = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, sharey=True)

    x1 = df1['Date'].values
    y1 = df1['Ct'].values

    x2 = df2['Date'].values
    y2 = df2['Ct'].values

    axes1[0].scatter(x1, y1)
    axes1[0].set_title(host1 + ' Ct values')

    axes1[0].set_xlabel('Date')
    axes1[0].set_ylabel('Ct')

    axes1[1].scatter(x2, y2)
    axes1[1].set_title(host2 + ' Ct values')

    axes1[1].set_xlabel('Date')
    axes1[1].set_ylabel('Ct')

    fig1.autofmt_xdate()

    fig1.savefig('./figures/Scatter_seasons.png', format='png', dpi=300, transparent=False)

    #fig2, axes2 = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, sharey=True)
    fig2, axes2 = plt.subplots(figsize=(12, 6), nrows=1, ncols=1)
    data1_low = df1[df1['Ct_Group'] == 0]
    data1_high = df1[df1['Ct_Group'] == 1]
    data2_low = df2[df2['Ct_Group'] == 0]
    data2_high = df2[df2['Ct_Group'] == 1]

    axes2.boxplot([data1_low['Ct'], data1_high['Ct']], labels=['low', 'high'])
    axes2.set_title(host1 + ' Ct values boxplot per High/Low group')
    axes2.set_xlabel('Group')
    axes2.set_ylabel('Ct');

    # axes2[1].boxplot([data2_low['Ct'], data2_high['Ct']], labels=['low', 'high'])
    # axes2[1].set_title(host2 + ' Ct values boxplot per High/Low group')
    # axes2[1].set_xlabel('Group')
    # axes2[1].set_ylabel('Ct');

    fig2.savefig('./figures/BoxPlot_highlow.png', format='png', dpi=300, transparent=False)

    fig3, axes3 = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, sharey=True)
    sns.violinplot(x="Host", y="Ct", data=df1, ax=axes3[0], inner="box")
    sns.violinplot(x="Host", y="Ct", data=df2, ax=axes3[1], inner="box")

    axes3[0].set_title(host1 + ' Violin plot');
    axes3[0].set_xlabel('Samples dispersion by Ct value')
    axes3[0].set_ylabel('Ct value')

    axes3[1].set_title(host2 + ' Violin plot');
    axes3[1].set_xlabel('Samples dispersion by Ct value')
    axes3[1].set_ylabel('Ct value')

    fig3.savefig('./figures/ViolinPlot_highlow.png', format='png', dpi=300, transparent=False)
# %%
plot_figures_2('Callithrix', 'Alouatta', callithrix_df, alouatta_df)
# %%

# %%
def plot_figures_3(host1, host2, df1, df2):
    fig1, axes1 = plt.subplots(figsize=(12, 6), nrows=2, ncols=1, sharex=True)

    x1 = df1['Date'].values
    y1 = df1['Ct'].values

    x2 = df2['Date'].values
    y2 = df2['Ct'].values

    axes1[0].scatter(x1, y1)
    axes1[0].set_title(host1 + ' Ct values')

    axes1[0].set_xlabel('Sample Collection Date')
    axes1[0].set_ylabel('Ct value by sample')

    axes1[0].set_ylim((0, 40))

    axes1[1].scatter(x2, y2)
    axes1[1].set_title(host2 + ' Ct values')

    axes1[1].set_xlabel('Sample Collection Date')
    axes1[1].set_ylabel('Ct value by sample')

    axes1[1].set_ylim((0, 40))

    fig1.autofmt_xdate()

    fig1.savefig('./figures/Scatter2_seasons.png', format='png', dpi=300, transparent=False)
# %%
plot_figures_3('Callithrix', 'Alouatta', callithrix_df, alouatta_df)



metadata.to_csv("../Callithrix_Analysis/DATA/!CLEAN/metadata_YFV_NHP_LatLon.csv")
