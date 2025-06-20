import os
import pdal
import json
import shapely
import numpy as np
import geopandas as gpd

class Calculate:
    def __init__(self):
        self.ontario_forests = gpd.read_file("additional/ontario_forests_dir/FRI_Tile_Index.shp")
        self.ontario_trails = gpd.read_file("additional/ontario_parks_dir/Ontario_Trail_Network_(OTN)_Segment.shp")
        self.ontario_parks = gpd.read_file("additional/ontario_trails_dir/PROV_PARK_REGULATED.shp")
        for i in [self.ontario_forests, self.ontario_parks, self.ontario_trails]:
            i.to_crs(self.ontario_forests.crs)

        self.selected_trail = None
        self.selected_park = None

        print(self.ontario_forests.crs)
        print(self.ontario_parks.columns)
        print(self.ontario_trails.columns)

    def select_trail(self, name=None, park=None, bbox=None):
        if name is not None:
            self.ontario_trails = self.ontario_trails[self.ontario_trails["TRAIL_NAME"].str.contains(name, case=False, na=False)]
            print(self.ontario_trails)
        elif park is not None:
            self.ontario_parks = self.ontario_parks[self.ontario_parks.str.contains(name, case=False, na=False)]


def main():
    pipeline = [
        {
            "type": "readers.las",
            "filename": "your_file.copc.laz",
            "spatialreference": "EPSG:32617"  # change to match your data
        },
        {
            "type": "filters.crop",
            "bounds": "([500000, 501000],[5100000,5101000])"  # your AOI bbox in map units
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"  # class 2 = ground
        },
        {
            "type": "writers.gdal",
            "filename": "dtm.tif",
            "resolution": 1.0,
            "output_type": "min",  # min Z value per pixel (ground surface)
            "data_type": "float"
        }
    ]

    p = pdal.Pipeline(json.dumps(pipeline))
    p.execute()

c = Calculate()
#c.select_trail("Algonquin Provincial Park Canoe Routes")

test = gpd.read_file("additional/jeffs_maps/campsites.shp")
print(test.columns)