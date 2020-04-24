#!/usr/bin/env python3

"""
- Create functions to get a reference genome and a dataset of aligned genomes.
- Search where in the reference genome this dataset starts.
 - use regex to do it.
- Also get a reference polyprotein record, associated with the reference genome.
- Make a mapping between a position in the genome
and an aminoacid in the polyprotein.
- Make a dict to keep {gen_pos_range : protein}

- ALL POSITIONS START AT ZERO!!! 000000000000


Duplicate line:                             Cmd + Shift + D
Move the current line Up or Down:           Cmd + Ctrl + Up
Select the next matching characters:        Cmd + D
Unselect                                    Cmd + U
Select all matching characters              Cmd + Ctrl + G
Toggle comments                             Cmd + /

"""

import Bio
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Alphabet import IUPAC

import pandas as pd
import numpy as np
import re
import datetime

"""
#######################################################################

#######################################################################
"""

def read_data(ref_genome_file, ref_polyprot_file, querry_seq_file):
    """
    Reads data into Bio SeqRecord
    """
    genome = SeqIO.read(ref_genome_file, "genbank")
    polyprot = SeqIO.read(ref_polyprot_file, "genbank")
    querry = SeqIO.parse(querry_seq_file, "fasta")
    first_record = next(querry)
    querry_seq = first_record.seq

    return (genome, polyprot, querry_seq)

def find_align_start(seq, ref_genome, search_size):
    """
    Given a dataset (multifasta alignment in '.aln' clustal format)
    and a reference genbank file (in '.gb' format), finds
    where in the reference sequence the dataset starts.
    It uses regex to find the initial sequence pattern in the reference genome.
    `search_size` is the number of nucleotides to include in the regex.
    Returns the position relative to the reference.
    """
    ref = str(ref_genome.seq)

    regex = re.compile(str(seq[:search_size]))

    ref_location = regex.search(ref)
    start_pos_ref = ref_location.start()
    return start_pos_ref


def read_polyprotein(ref_polyprot):
    """
    Given a polyprotein genbank file (in '.gp' format), parses through
    its features and returns a dictionary containing the proteins
    names as keys and positions (start:end in "biopython location") as values.
    """
    dic_prot = {}

    for feature in ref_polyprot.features:
        if 'region_name' in feature.qualifiers:
            value = (feature.location)
            key = (feature.qualifiers['region_name'][0])
            dic_prot[key] = value
    return dic_prot


def pos_aminoacid(nn_pos, seq_rel_start, ref_genome, ref_polyprot):
    """
    nn_pos: nucleotide position in the dataset sequence.
    seq_rel_start: position where the dataset starts in the reference genome.

    Given a nucleotide position (int) and a reference genome (".gb" format),
    returns:

    aa_pos: Which aminoacid position in the translated polyprotein
    it is in.
    aa: The translated amino acid.
    codon: The codon in the reference genome.
    codon_pos: The codon position (0, 1, 2).
    """
    for feature in ref_genome.features:
        if feature.type == "CDS":
            cds_start = int(feature.location.start)
            break

    # position in the reference genome CDS, relative to its CDS start.
    cds_pos = (nn_pos + seq_rel_start) - cds_start

    # aminoacid position in the polyprotein
    aa_pos = (cds_pos) // 3

    # nucleotide position inside codon (0, 1, 2)
    codon_pos = (cds_pos) % 3

    # translated aminoacid
    aa = ref_polyprot.seq[aa_pos]

    # codon starting position in the dataset sequence
    codon_start = nn_pos - codon_pos

    # codon start pos in the reference genome.
    ref_codon_start_pos = codon_start + seq_rel_start
    # three letter codon
    codon = ref_genome.seq[ref_codon_start_pos:ref_codon_start_pos+3]

    return (aa_pos, aa, codon, codon_pos)


def seq_snv_info(nn_pos, seq, ref_genome, ref_polyprot, search_size=20):
    """
    given a position in a database sequence, returns:
    - sequence codon
    - sequence aminoacid
    - reference codon
    - reference aminoacid
    - codon position
    """

    seq_rel_start = find_align_start(seq, ref_genome, search_size)
    # codon start pos in the reference genome.

    for feature in ref_genome.features:
        if feature.type == "CDS":
            cds_start = int(feature.location.start)
            break

    # position in the reference genome relatice to its start.
    ref_pos = (nn_pos + seq_rel_start)
    # position in the reference genome CDS, relative to its CDS start.
    cds_pos = (nn_pos + seq_rel_start) - cds_start
    # aminoacid position in the polyprotein
    aa_pos = (cds_pos) // 3
    # nucleotide position inside codon (0, 1, 2)
    codon_pos = (cds_pos) % 3

    # codon starting position in the dataset sequence
    codon_start = nn_pos - codon_pos
    # three letter codon
    codon_seq = seq[codon_start:codon_start+3]
    aa_seq = codon_seq.translate()

    ref_codon_start_pos = codon_start + seq_rel_start
    # three letter codon
    codon_ref = ref_genome.seq[ref_codon_start_pos:ref_codon_start_pos+3]
    # translated aminoacid
    aa_ref = ref_polyprot.seq[aa_pos]

    return (codon_seq, aa_seq, ref_pos, codon_ref, aa_ref, codon_pos)

def which_protein(nn_snv, aa_pos, dic_prot, case):
    """
    Given an aminoacid position, returns in which protein inside the
    polyprotein it is.
    """
    print("SNV at position {} on genomes under analysis\n".format(nn_snv))

    log_file = './OUTPUT/log_{}_SNV_INFO.txt'.format(case)

    for prot in dic_prot:
        if aa_pos in dic_prot[prot]:

            print(prot)
            start = int(dic_prot[prot].start)
            print('Protein start in reference polyprotein: {}'.format(start))
            print('Residue "{}" position in reference polyprotein: {}'.format(aa_ref, aa_pos))
            pos_in_protein = aa_pos - start
            print('Residue "{}" position in {} protein: {}'.format(aa_ref, prot, pos_in_protein))
            neighbours = ref_polyprot.seq[aa_pos-5 : aa_pos +6]
            print('Neighbouring residues ("{}" is in the middle): {}'.format(aa_ref, neighbours))

            with open(log_file, 'a') as log:
                x = datetime.datetime.now()
                log.write('SNV info:\n{0}\n\n'.format(x))
                log.write("SNV at position {} on genomes under analysis.\n".format(nn_snv))
                log.write('Protein: {}.\n'.format(prot))
                log.write('Protein start in reference polyprotein: {}.\n'.format(start))
                log.write('Residue "{}" position in reference polyprotein: {}.\n'.format(aa_ref, aa_pos))
                log.write('Residue "{}" position in {} protein: {}.\n'.format(aa_ref, prot, pos_in_protein))
                log.write('Neighbouring residues ("{}" is in the middle): {}.\n'.format(aa_ref, neighbours))
                log.write('////////////////////////////////////////////////\n\n')


            return prot
            break



"""
#######################################################################
MAIN
#######################################################################
"""

if __name__ == "__main__":

    #%%
    # NOVO

    # First I read the sequences I'll be working with.
    # Even though both references are in GB format, which contains a lot of
    # metadata info, SeqIO.read knows to separate it neatly so I can get
    # the nucleotide or aminoacid sequence just by calling its "seq" attribute.

    # The querry sequence must be one of the sequences I'm actually analizing.
    # It must be the one that starts first and, preferably, with no gaps in
    # its first 10 to 20 positions.

    # IDEA for automatically finding a suitable querry inside a dataset:

    # for seq in fasta:
    #     for position in seq:
    #         if nucleotide is not gap:
    #             lowest_position = position
    #     if lowest_position < best_lowest_position:
    #         if nucleotides from lowest_position to lowest_position+10 are not gap:
    #             best_lowest_position = lowest_position
    #             best_seq = seq

    case = "HUMAN"

    ref_genome_file = '../DATA/Reference_GENBANK_YFV/YFV_BeH655417_JF912190.gb'
    ref_polyprot_file = '../DATA/Reference_GENBANK_YFV/YFV_BeH655417_JF912190_polyprot.gb'
    querry_seq_file = '../DATA/Human_Analisys/DATA/querry_seq_marielton.fas'

    ref_genome = SeqIO.read(ref_genome_file, "genbank")
    ref_polyprot = SeqIO.read(ref_polyprot_file, "genbank")
    querry = SeqIO.parse(querry_seq_file, "fasta")
    first_record = next(querry)
    querry_seq = first_record.seq
    #%%
    # (ref_genome, ref_polyprot, querry_seq) = read_data(ref_genome_file, ref_polyprot_file, querry_seq_file)

    seq_rel_start = find_align_start(querry_seq, ref_genome, 10)
    print(seq_rel_start)
    s1 = querry_seq[:20]
    s2 = ref_genome[seq_rel_start:seq_rel_start+20].seq
    s1 == s2

    dic_prot = read_polyprotein(ref_polyprot)

    nn_snv = 1000

    (aa_pos, aa, codon, codon_pos) = pos_aminoacid(nn_snv, seq_rel_start, ref_genome, ref_polyprot)

    print(aa_pos)
    print(aa)
    print(codon)
    print(codon_pos)

    prot = which_protein(nn_snv, aa_pos, dic_prot, case)
    print(prot)


    (codon_seq, aa_seq, ref_pos, codon_ref, aa_ref, codon_pos) = seq_snv_info(1000, querry_seq, ref_genome, ref_polyprot)

    codon_seq
    aa_seq
    codon_ref
    aa_ref
    codon_pos
    ref_pos

    ref_polyprot.seq[aa_pos]

"""
#######################################################################

#######################################################################
"""
