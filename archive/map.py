#!/usr/bin/env python3


import pandas as pd
import geopandas
import matplotlib.pyplot as plt

# First, let’s consider a DataFrame containing cities and their respective longitudes and latitudes.
df = pd.DataFrame(
    {'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
     'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
     'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48],
     'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]})

# A GeoDataFrame needs a shapely object. We use geopandas points_from_xy() to transform Longitude and Latitude into a list of shapely.Point objects and set it as a geometry while creating the GeoDataFrame. (note that points_from_xy() is an enhanced wrapper for [Point(x, y) for x, y in zip(df.Longitude, df.Latitude)])

gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))

print(gdf)

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
sa = world[world['continent'] == 'South America']
type(sa)
dir(sa)
# We restrict to South America.

fig, ax = plt.subplots()
ax = sa.plot(color='white', edgecolor='black')

# We can now plot our GeoDataFrame.
gdf.plot(ax=ax, color='red')

plt.show()
