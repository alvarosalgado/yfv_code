#!/usr/bin/env python3

import pandas as pd

lat_lon = pd.read_csv("lat_long.txt", header=None, names=["ID", "Location", "LAT", "LON"])

lat_lon.shape
