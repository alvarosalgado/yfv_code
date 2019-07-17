#!/usr/bin/env python3

"""
Prints the visualization of a genome
"""
from dna_features_viewer import GraphicFeature, GraphicRecord
features=[
    GraphicFeature(start=0, end=20, strand=+1, color="#ffd700",
                   label="Small feature"),
    GraphicFeature(start=20, end=500, strand=+1, color="#ffcccc",
                   label="Gene 1 with a very long name"),
    GraphicFeature(start=400, end=700, strand=-1, color="#cffccc",
                   label="Gene 2"),
    GraphicFeature(start=600, end=900, strand=+1, color="#ccccff",
                   label="Gene 3")
]
record = GraphicRecord(sequence_length=1000, features=features)
record.plot(figure_width=5)

from dna_features_viewer import BiopythonTranslator
graphic_record = BiopythonTranslator().translate_record("../Callithrix_Analysis/DATA/!CLEAN/YFV_polyprotein_AFH35044.gb")

ax, _ = graphic_record.plot(figure_width=10)

graphic_record.features = graphic_record.features[2:-1]

ax, _ = graphic_record.plot(figure_width=10)
dir(graphic_record.features[0])
graphic_record.features[0].label = "capsid"
graphic_record.features[0].color = "red"

graphic_record.features[1].label = "propep"
graphic_record.features[1].color = "blue"

graphic_record.features[2].label = "M"
graphic_record.features[2].color = "pink"

graphic_record.features[3].label = "propep"
graphic_record.features[3].color = "blue"

graphic_record.features[4].label = "glycoprotein"
graphic_record.features[4].color = "yellow"

graphic_record.features[5].label = "glycop_C"
graphic_record.features[5].color = "green"

graphic_record.features[6].label = "E_stem"
graphic_record.features[6].color = "blue"

graphic_record.features[7].label = "NS1"
graphic_record.features[7].color = "gray"

graphic_record.features[8].label = "NS2A"
graphic_record.features[8].color = "red"

graphic_record.features[9].label = "NS2B"
graphic_record.features[9].color = "pink"

graphic_record.features[10].label = "S7"
graphic_record.features[10].color = "red"

graphic_record.features[11].label = "DEXDC"
graphic_record.features[11].color = "red"

graphic_record.features[12].label = "DEAD"
graphic_record.features[12].color = "red"

graphic_record.features[13].label = "other"
graphic_record.features[13].color = "red"
