


import numpy as np
import pandas as pd
from scipy.stats import entropy
from math import log, e


from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord
import ref_genome_polyprot_toolbox as g_tool
from reportlab.lib import colors
from reportlab.lib.units import cm
from Bio.Graphics import GenomeDiagram
from Bio import SeqIO

import os, sys


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


case = "HUMAN_YFV"

ALL_fasta_filename = data_dir+"/ALN_YFV_ALL_HUMAN+REF.fasta"
ref_genome_file = working_dir+'/3_REFERENCE/YFV_REF_NC_002031.fasta'
ref_polyprot_file = working_dir+'/3_REFERENCE/EDITED_YFV_REF_NC_002031.gb'

ref_genome = SeqIO.read(ref_genome_file, "fasta").lower()
ref_polyprot = SeqIO.read(ref_polyprot_file, "genbank").lower()

# Get analysis dataset to draw other info, such as variability or coverage.
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(ALL_fasta_filename, "fasta")]
description = [seq_rec.description for seq_rec in SeqIO.parse(ALL_fasta_filename, "fasta")]
seqs = np.array([list(str(seq_rec.seq.lower())) for seq_rec in SeqIO.parse(ALL_fasta_filename, "fasta")])
seqs.shape
cols = np.array(list(range(seqs.shape[1])))
cols = cols + 1

seq_df = pd.DataFrame(seqs, index=identifiers, columns=cols)
seq_df = seq_df.loc[~seq_df.index.duplicated(keep='first')]

# seq_df = seq_df.copy()

#%%
# Trim sequences, because FUNED seqs are shorter than the rest.

seq_for_trim = seq_df.iloc[2]

indexa = int(np.where((seq_for_trim=='a'))[0][0])+1
indexc = int(np.where((seq_for_trim=='c'))[0][0])+1
indext = int(np.where((seq_for_trim=='t'))[0][0])+1
indexg = int(np.where((seq_for_trim=='g'))[0][0])+1
start = np.min([indexa, indexc, indext, indexg])
seq_for_trim[start]

indexa = int(np.where((seq_for_trim=='a'))[0][-1])+1
indexc = int(np.where((seq_for_trim=='c'))[0][-1])+1
indext = int(np.where((seq_for_trim=='t'))[0][-1])+1
indexg = int(np.where((seq_for_trim=='g'))[0][-1])+1
finish = np.max([indexa, indexc, indext, indexg])
seq_for_trim[finish]

lc = int(seq_df.columns[-1])

last_colums = np.arange(finish+1, lc+1)
first_columns = np.arange(1,start)

seq_df[first_columns]
seq_df[last_colums]

seq_df.drop(last_colums, axis=1, inplace=True)
seq_df.drop(first_columns, axis=1, inplace=True)

#Remove first sequence, used only for column numbering
seq_df.drop(seq_df.index[0], axis=0, inplace=True)
seq_df.drop(seq_df.index[0], axis=0, inplace=True)


#%%
# Replace not-sequenced positions with np.nan

unique = pd.unique(seq_df.values.ravel('K'))

# Get a list of all characters other than "actg" are present in the DataFrame
unique = unique[unique!='a']
unique = unique[unique!='c']
unique = unique[unique!='t']
unique = unique[unique!='g']
to_replace = unique

# Replace them with np.nan
seq_df.replace(to_replace, np.nan, inplace=True)

#%%
# Create list of tuples containing (position, entropy)

def entropy1(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

alignment_entropy=[]

for col in seq_df.columns:
    bases = seq_df[col]
    bases.dropna(how='any', inplace=True)
    ent = entropy1(bases)
    pos_ent = (col, ent)
    alignment_entropy.append(pos_ent)


#%%

for feature in ref_polyprot.features:
    print(feature)

# Bottom Up Approach

feature_set1 = GenomeDiagram.FeatureSet()
feature_set2 = GenomeDiagram.FeatureSet()

i=1
for feature in ref_polyprot.features:
    if feature.type != "mat_peptide":
        #Exclude this feature
        continue
    if i % 2 == 0:
        feature_set1.add_feature(feature,
            color='green',
            label=True,
            label_size=14,
            label_angle=90,
            label_position='start',
            label_strand=1,
            strand=None)
        # feature_set_labels_top.add_feature(feature,
        #     color='green',
        #     label=True,
        #     label_size=14,
        #     label_angle=90,
        #     label_position='start',
        #     label_strand=1,
        #     strand=None)
    else:
        feature_set2.add_feature(feature,
            color='coral',
            label=True,
            label_size=14,
            label_angle=270,
            label_position='end',
            label_strand=-1,
            strand=None)
    i += 1


genes_track = GenomeDiagram.Track('genes',
    greytrack=False,
    scale=False)



genes_track.add_set(feature_set1)
genes_track.add_set(feature_set2)





#%%
from Bio.SeqFeature import SeqFeature, FeatureLocation

snv_df = pd.read_csv(tab_dir+'/SNV_HUMAN_YFV_RESULTS.csv')
snv_series = snv_df.iloc[:, 2]
feature_set_SNV = GenomeDiagram.FeatureSet()

for position in snv_series:
    snv = SeqFeature(FeatureLocation(position,position),strand=+1)
    feature_set_SNV.add_feature(snv, color='red', strand=None)

SNV_track = GenomeDiagram.Track('SNV',
    greytrack=False,
    scale=True,
    scale_format='SInt',
    scale_fontsize = 10,
    scale_fontangle = 90,
    scale_largetick_interval=5000,
    scale_smalltick_interval=1000,
    scale_largeticks=0.5,
    scale_smallticks=0.2)

SNV_track.add_set(feature_set_SNV)


#%%

entropy_graph_set = GenomeDiagram.GraphSet('entropy')
entropy_graph = entropy_graph_set.new_graph(alignment_entropy, name='entropy', style='line')

entropy_track = GenomeDiagram.Track('entropy')
entropy_track.add_set(entropy_graph_set)




HUMAN_YFV_diagram = GenomeDiagram.Diagram("YFV Human Infections",
    pagesize='A4',
    orientation='portrait',
    x=0.2,
    y=0.3,
    tracklines=False)
HUMAN_YFV_diagram.add_track(genes_track, 1)
HUMAN_YFV_diagram.add_track(entropy_track, 2)
HUMAN_YFV_diagram.add_track(SNV_track, 3)

HUMAN_YFV_diagram.draw(format="linear", orientation="landscape", pagesize='A4', fragments=1, start=0, end=len(ref_polyprot))
HUMAN_YFV_diagram.write(fig_dir+"/HUMAN_YFV_diagram.svg", "SVG")










"""///////////////////////////////////////////////////////////////////////"""
"""///////////////////////////////////////////////////////////////////////"""
"""///////////////////////////////////////////////////////////////////////"""

''' NHP '''

"""///////////////////////////////////////////////////////////////////////"""
"""///////////////////////////////////////////////////////////////////////"""
"""///////////////////////////////////////////////////////////////////////"""


""" /////////////////////////////////////////////////////////////////////// """

""" SET THE STAGE... """

# Create OUTPUT dir inside DATA dir, where all processed data, figures, tbles, ect will be stored

working_dir = '/Users/alvarosalgado/Google Drive/Bioinformática/!Qualificação_alvaro/YFV'

if os.path.isdir(working_dir+'/2_OUTPUT/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/')

if os.path.isdir(working_dir+'/2_OUTPUT/NHP/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/NHP/')

if os.path.isdir(working_dir+'/2_OUTPUT/NHP/FIGURES/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/NHP/FIGURES/')
if os.path.isdir(working_dir+'/2_OUTPUT/NHP/TABLES/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/NHP/TABLES/')
if os.path.isdir(working_dir+'/2_OUTPUT/NHP/PICKLE/')==False:
    os.mkdir(working_dir+'/2_OUTPUT/NHP/PICKLE/')

out_dir = working_dir+'/2_OUTPUT/NHP'
fig_dir = working_dir+'/2_OUTPUT/NHP/FIGURES'
tab_dir = working_dir+'/2_OUTPUT/NHP/TABLES'
pik_dir = working_dir+'/2_OUTPUT/NHP/PICKLE'
data_dir = working_dir+'/1_DATA/Callithrix_Analysis/DATA/!CLEAN'


case = "NHP_YFV"

ALL_fasta_filename = data_dir+"/YFV_NHP_ALL_REF_ALN.fasta"
ref_genome_file = working_dir+'/3_REFERENCE/YFV_REF_NC_002031.fasta'
ref_polyprot_file = working_dir+'/3_REFERENCE/EDITED_YFV_REF_NC_002031.gb'

ref_genome = SeqIO.read(ref_genome_file, "fasta").lower()
ref_polyprot = SeqIO.read(ref_polyprot_file, "genbank").lower()

# Get analysis dataset to draw other info, such as variability or coverage.
identifiers = [seq_rec.id for seq_rec in SeqIO.parse(ALL_fasta_filename, "fasta")]
description = [seq_rec.description for seq_rec in SeqIO.parse(ALL_fasta_filename, "fasta")]
seqs = np.array([list(str(seq_rec.seq.lower())) for seq_rec in SeqIO.parse(ALL_fasta_filename, "fasta")])
seqs.shape
cols = np.array(list(range(seqs.shape[1])))
cols = cols + 1

seq_df = pd.DataFrame(seqs, index=identifiers, columns=cols)
seq_df = seq_df.loc[~seq_df.index.duplicated(keep='first')]

# seq_df = seq_df.copy()

#%%
# Trim sequences, because FUNED seqs are shorter than the rest.

seq_for_trim = seq_df.iloc[2]

indexa = int(np.where((seq_for_trim=='a'))[0][0])+1
indexc = int(np.where((seq_for_trim=='c'))[0][0])+1
indext = int(np.where((seq_for_trim=='t'))[0][0])+1
indexg = int(np.where((seq_for_trim=='g'))[0][0])+1
start = np.min([indexa, indexc, indext, indexg])
seq_for_trim[start]

indexa = int(np.where((seq_for_trim=='a'))[0][-1])+1
indexc = int(np.where((seq_for_trim=='c'))[0][-1])+1
indext = int(np.where((seq_for_trim=='t'))[0][-1])+1
indexg = int(np.where((seq_for_trim=='g'))[0][-1])+1
finish = np.max([indexa, indexc, indext, indexg])
seq_for_trim[finish]

lc = int(seq_df.columns[-1])

last_colums = np.arange(finish+1, lc+1)
first_columns = np.arange(1,start)

seq_df[first_columns]
seq_df[last_colums]

seq_df.drop(last_colums, axis=1, inplace=True)
seq_df.drop(first_columns, axis=1, inplace=True)

#Remove first sequence, used only for column numbering
seq_df.drop(seq_df.index[0], axis=0, inplace=True)


#%%
# Replace not-sequenced positions with np.nan

unique = pd.unique(seq_df.values.ravel('K'))

# Get a list of all characters other than "actg" are present in the DataFrame
unique = unique[unique!='a']
unique = unique[unique!='c']
unique = unique[unique!='t']
unique = unique[unique!='g']
to_replace = unique

# Replace them with np.nan
seq_df.replace(to_replace, np.nan, inplace=True)

#%%
# Create list of tuples containing (position, entropy)

def entropy1(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

alignment_entropy=[]

for col in seq_df.columns:
    bases = seq_df[col]
    bases.dropna(how='any', inplace=True)
    ent = entropy1(bases)
    pos_ent = (col, ent)
    alignment_entropy.append(pos_ent)


#%%

for feature in ref_polyprot.features:
    print(feature)

# Bottom Up Approach

feature_set1 = GenomeDiagram.FeatureSet()
feature_set2 = GenomeDiagram.FeatureSet()

i=1
for feature in ref_polyprot.features:
    if feature.type != "mat_peptide":
        #Exclude this feature
        continue
    if i % 2 == 0:
        feature_set1.add_feature(feature,
            color='green',
            label=True,
            label_size=14,
            label_angle=90,
            label_position='start',
            label_strand=1,
            strand=None)
        # feature_set_labels_top.add_feature(feature,
        #     color='green',
        #     label=True,
        #     label_size=14,
        #     label_angle=90,
        #     label_position='start',
        #     label_strand=1,
        #     strand=None)
    else:
        feature_set2.add_feature(feature,
            color='coral',
            label=True,
            label_size=14,
            label_angle=270,
            label_position='end',
            label_strand=-1,
            strand=None)
    i += 1


genes_track = GenomeDiagram.Track('genes',
    greytrack=False,
    scale=False)



genes_track.add_set(feature_set1)
genes_track.add_set(feature_set2)





#%%
from Bio.SeqFeature import SeqFeature, FeatureLocation

snv_df = pd.read_csv(tab_dir+'/SNV_NHP_YFV_RESULTS.csv')
snv_series = snv_df.iloc[:, 2]
feature_set_SNV = GenomeDiagram.FeatureSet()

for position in snv_series:
    snv = SeqFeature(FeatureLocation(position,position),strand=+1)
    feature_set_SNV.add_feature(snv, color='red', strand=None)

SNV_track = GenomeDiagram.Track('SNV',
    greytrack=False,
    scale=True,
    scale_format='SInt',
    scale_fontsize = 10,
    scale_fontangle = 90,
    scale_largetick_interval=5000,
    scale_smalltick_interval=1000,
    scale_largeticks=0.5,
    scale_smallticks=0.2)

SNV_track.add_set(feature_set_SNV)


#%%

entropy_graph_set = GenomeDiagram.GraphSet('entropy')
entropy_graph = entropy_graph_set.new_graph(alignment_entropy, name='entropy', style='line')

entropy_track = GenomeDiagram.Track('entropy')
entropy_track.add_set(entropy_graph_set)




HUMAN_YFV_diagram = GenomeDiagram.Diagram("YFV NHP Ct Investigation",
    pagesize='A4',
    orientation='portrait',
    x=0.2,
    y=0.3,
    tracklines=False)
HUMAN_YFV_diagram.add_track(genes_track, 1)
HUMAN_YFV_diagram.add_track(entropy_track, 2)
HUMAN_YFV_diagram.add_track(SNV_track, 3)

HUMAN_YFV_diagram.draw(format="linear", orientation="landscape", pagesize='A4', fragments=1, start=0, end=len(ref_polyprot))
HUMAN_YFV_diagram.write(fig_dir+"/NHP_YFV_diagram.svg", "SVG")
