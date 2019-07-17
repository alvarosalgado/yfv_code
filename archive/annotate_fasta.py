#!/usr/bin/env python3
# author: Álvaro Salgado
# email: salgado.alvaro@me.com
# description: Edit fasta identifier, including an annotation in the beginning.

import sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC

def sequence_annotator(fasta_file, note):

    # Create a list with sequences
    records = list(SeqIO.parse(fasta_file, "fasta"))
    new_records = []

    # Iterate over list and change id's
    for seq_record in records:
        # Take the current sequence
        seq = str(seq_record.seq).upper()
        id = seq_record.id
        name = seq_record.name
        description = seq_record.description

        # Edit sequences to UPPERCASE and IUPAC alphabet
        # Append note to beginning of id
        new_seq = Seq(seq)
        #new_seq = Seq(seq, alphabet=IUPAC.unambiguous_dna)
        new_id = note + '|' + id

        # Create "SeqRecord" object
        new_seq_record = SeqRecord(new_seq, id=new_id, name=name,
        description=description)
        # Insert SeqRecord in new list
        new_records.append(new_seq_record)

    # Write the annotated sequences list to file
    SeqIO.write(new_records, "annotated_" + fasta_file, "fasta")

    print("ANNOTATED!!!\nPlease check annotated_" + fasta_file)


userParameters = sys.argv[1:]

try:
    if len(userParameters) == 2:
        sequence_annotator(userParameters[0], str(userParameters[1]))
    else:
        print("There is a problem!")
except:
    print("There is a problem!")
