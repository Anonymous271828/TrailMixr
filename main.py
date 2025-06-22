import os
import pdal
import json
import shapely
import requests
import numpy as np
import geopandas as gpd
from itertools import combinations

API_KEY = "kLyjHKWOs4w8hDq4PFc7QZxPyF1OgNRh"

class Calculate:

    DISTANCE_THRESH = 40
    DISTANCE_THRESH_FOR_EVENTS = 40

    def __init__(self):
        self.ontario_forests = gpd.read_file("additional/ontario_forests_dir/FRI_Tile_Index.shp").to_crs(32617)
        self.ontario_trails = gpd.read_file("additional/Non_Sensitive.gdb").to_crs(32617)
        self.ontario_parks = gpd.read_file("additional/ontario_trails_dir/PROV_PARK_REGULATED.shp").to_crs(32617)
        self.algonquin_campsites = gpd.read_file("additional/jeffs_maps/campsites.shp").to_crs(32617)
        self.topography = gpd.read_file("additional/elevation/ONT_ELEVATION_DATA_INDEX.shp").to_crs(32617)
        self.events = gpd.read_file("additional/jeffs_maps/campsites.shp").to_crs(32617).translate(xoff=100, yoff=-100)

        self.days = max(0, 5)

        self.selected_trail = None
        self.selected_park = None
        self.selected_campsites = []

        self.schedule = []
        self.lat, self.long = None, None
        self.combined = []


    def select_trail(self, name=None, park=None, bbox=None):
        if name is not None:
            self.selected_trail = self.ontario_trails[self.ontario_trails["TRAIL_NAME"].str.contains(name, case=False, na=False)]
            self.lat, self.long = self.selected_trail.to_crs(4326).geometry.centroid.y, self.selected_trail.to_crs(4326).geometry.centroid.x
            trail_line = self.selected_trail.union_all()
            distance = self.algonquin_campsites.geometry.apply(lambda pt: self.selected_trail.distance(pt))
            numeric_cols = list(distance.select_dtypes(include=[np.number]).columns)
            distance = distance[distance[numeric_cols[0]] < self.DISTANCE_THRESH]
            
            matched = self.algonquin_campsites.loc[self.algonquin_campsites.index.isin(distance.index)]
            if not matched.empty:
                distance_along_path = matched.centroid.apply(lambda pt: trail_line.project(pt))
            else:
                distance_along_path = None

            distance_along_path = distance_along_path.sort_values(ascending=True)
            indexes = distance_along_path.index.tolist()
            for i, j in enumerate(distance_along_path):
                self.selected_campsites.append(Campsite(indexes[i], j, distance.loc[indexes[i]]))

        elif park is not None:
            self.ontario_parks = self.ontario_parks[self.ontario_parks["PARK_NAME"].str.contains(name, case=False, na=False)]

    def organize_events(self, events):
        # FOR WHEN WE ACTUALLY GET A DATASET
        #self.events = self.events[self.events["TRAIL_NAME"].str.contains("|".join(events), case=False, na=False)]
        
        trail_line = self.selected_trail.union_all()
        distance = self.events.geometry.apply(lambda pt: self.selected_trail.distance(pt))
        numeric_cols = list(distance.select_dtypes(include=[np.number]).columns)
        distance = distance[distance[numeric_cols[0]] < self.DISTANCE_THRESH_FOR_EVENTS]
        
        matched = self.events.loc[self.events.index.isin(distance.index)]
        if not matched.empty:
            distance_along_path = matched.centroid.apply(lambda pt: trail_line.project(pt))
        else:
            distance_along_path = None

        distance_along_path = distance_along_path.sort_values(ascending=True)
        
        indexes = distance_along_path.index.tolist()
        self.events = []
        for i, j in enumerate(distance_along_path):
            self.events.append(Event(indexes[i], j, distance.loc[indexes[i]]))
        
    
    def recurse_campsites(self, trail, days):
        if days <= 0:
            return []
        else:
            for i in range(days):
                temp_weather = Weather(i, self.lat, self.long)
                self.schedule.append(
                    Day(
                        temp_weather.find_sunset(), 
                        temp_weather.find_sunrise(), 
                        temp_weather.score_each_hour()
                        ))
            for i in combinations(self.selected_campsites, days):
                for a,b in enumerate(i):
                    self.schedule[a].stop = b
                ### Code here
                    weather = Weather(a, self.lat, self.long, self.extract_legs(trail))
                    self.schedule[a].change_hour(1, True)

    def extract_legs(self, trail):
        if isinstance(trail, shapely.geometry.LineString):
            return [[pt[1], pt[0]] for pt in trail.coords]
        elif isinstance(trail, shapely.geometry.MultiLineString):
            legs = []
            for line in trail:
                legs.extend([[pt[1], pt[0]] for pt in line.coords])
            return legs
        else:
            raise TypeError("Geometry must be LineString or MultiLineString")
        
    def extract_data(self):
        buffered_trail = self.selected_trail.buffer(10)

        pipeline = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": "/Volumes/STORAGES/test.copc.laz"
                },
                {
                    "type": "filters.crop",
                    "polygon": buffered_trail.unary_union.wkt
                },
                {
                    "type": "filters.range",
                    "limits": "Classification[1:7]"
                },
                {
                    "type": "writers.las",
                    "filename": "trail_buffered.laz"
                }
            ]
        }

        pipeline_json = json.dumps(pipeline)
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()
        
        arrays = pipeline.arrays[0]
        print(arrays)
        veg_points = arrays[np.isin(arrays['Classification'], [4, 5])]
        low_veg_points = arrays[arrays['Classification'] == 3]

        x = veg_points['X']
        y = veg_points['Y']

        res = 1
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_bins = np.arange(x_min, x_max + res, res)
        y_bins = np.arange(y_min, y_max + res, res)


    def get_copc_laz(self):
        API_PATTERN = "https://download.fri.mnrf.gov.on.ca/api/api/Download/geohub/laz/utm16/1kmZ164040565102023L.copc.laz"
        intersection = self.ontario_forests[self.ontario_forests.intersects(self.selected_trail.union_all())]
        # for i in intersection["Tilename"]:
        #     if not os.path.isfile("additiona/forestry_map/{}.copc.laz".format(i)):
        #         requests.get("https://download.fri.mnrf.gov.on.ca/api/api/Download/geohub/laz/utm16/{}.copc.laz".format(i)).content

        
class Campsite:
    def __init__(self, index, distance_along_path, distance_from_trail):
        self.index = index
        self.distance_along_path = distance_along_path
        self.distance_from_trail = distance_from_trail

    def __repr__(self):
        return f"Campsite(index={self.index}, distance_along_path={self.distance_along_path}, distance_from_trail={self.distance_from_trail})"

class Day:
    def __init__(self, sunset, sunrise, weatherscore, events):
        self.hours = {1:False, 2:False, 3:False, 4:False, 5:False, 6:False, 7:False, 8:False, 9:False, 10:False, 11:False, 12:False, 13:False, 14:False, 15:False, 16:False, 17:False, 18:False, 19:False, 20:False, 21:False, 22:False, 23:False}
        self.weatherscore = weatherscore
        self.stop = None
    def change_hour(self, hour, val):
        if hour in self.hours:
            self.hours[hour] = val
        else:
            raise ValueError("Hour must be between 1 and 23")
        
    def get_hour(self, hour):
        return self.hours.get(hour, False)
    
class Event:
    def __init__(self, index, distance_along_path, distance_from_trail, hours=None, location=None):
        self.name = index
        self.id = id
        self.hours = hours
        self.location = location
    
class Weather:
    def __init__(self, date, lat, long, legs=None):
        self.url = "https://api.tomorrow.io/v4/timelines"
        self.date = date
        self.lat = lat
        self.long = long


        if legs is not None:
            trail_data = {
                "name": "Your Trail Name",
                "polyline": legs,
                "tags": ["Toronto", "Bike"]
            }

    def score_hour(self, v):
        score = 0.0
        v["temperature"] = float(v["temperature"])
        v["precipitationIntensity"] = float(v["precipitationIntensity"])
        v["windSpeed"] = float(v["windSpeed"])
        v["humidity"] = float(v["humidity"])
        v["cloudCover"] = float(v["cloudCover"])
        v["dewPoint"] = float(v["dewPoint"])
        v["visibility"] = float(v["visibility"])
        v["weatherCode"] = int(v["weatherCode"])

        score += max(0, 10 - abs(v["temperature"] - 20) * 0.5)

        score -= min(10, v["precipitationIntensity"] * 10)

        score -= min(5, v["windSpeed"] * 0.5)

        if v["humidity"] < 40:
            score -= (40 - v["humidity"]) * 0.1
        elif v["humidity"] > 60:
            score -= (v["humidity"] - 60) * 0.1

        score += max(0, 2 - abs(v["cloudCover"] - 50) / 25)

        diff = abs(v["temperature"] - v["dewPoint"])

        score += min(2, v["visibility"] / 10)

        clear = {1000, 1100, 1101}
        rain = {4000, 4200, 4210, 4201}
        snow = {5000, 5100, 5101}
        if v["weatherCode"] in clear:
            score += 2
        elif v["weatherCode"] in rain.union(snow):
            score -= 5

        return score
    
    def score_each_hour(self):
        params = {
            "location": f"{self.lat},{self.long}",  # Latitude,Longitude
            "fields": ["temperature", "precipitationIntensity", "windSpeed", "humidity", "cloudCover", "dewPoint", "visibility", "weatherCode"],
            "timesteps": "1h", 
            "units": "metric",
            "apikey": API_KEY 
        }
        response = requests.get(self.url, params=params)
        hourly_score = []
        data = response.json()
        for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]:
            v = data["data"]["timelines"][0]["intervals"][i]["values"]
            hourly_score.append(self.score_hour(v))

        return hourly_score

    def __repr__(self):
        return f"Weather(date={self.date}, temperature={self.temperature}, precipitation={self.precipitation}, wind_speed={self.wind_speed})"

    def find_sunrise(self):
        params = {
            "location": f"{self.lat},{self.long}",  # Latitude,Longitude
            "fields": "sunriseTime",
            "timesteps": "1d",  # Daily data
            "units": "metric",
            "apikey": API_KEY  # Replace with your actual API key
        }
        response = requests.get(self.url, params=params)
        data = response.json()

        sunrise_time = data["data"]["timelines"][0]["intervals"][0]["values"]["sunriseTime"]
        
        colon_loc = sunrise_time.index(":")
        if int(sunrise_time[colon_loc+1] + sunrise_time[colon_loc+2]) <= 30:
            sunrise_time = int(sunrise_time[colon_loc-1]) - 5
        else:
            sunrise_time = int(sunrise_time[colon_loc-1]) - 4

        return sunrise_time % 24

    def find_sunset(self):
        params = {
            "location": f"{self.lat},{self.long}",  # Latitude,Longitude
            "fields": "sunsetTime",
            "timesteps": "1d",  # Daily data
            "units": "metric",
            "apikey": API_KEY  # Replace with your actual API key
        }
        response = requests.get(self.url, params=params)
        data = response.json()

        sunset_time = data["data"]["timelines"][0]["intervals"][0]["values"]["sunsetTime"]
        colon_loc = sunset_time.index(":")
        if int(sunset_time[colon_loc+1] + sunset_time[colon_loc+2]) <= 30:
            sunset_time = int(sunset_time[colon_loc-1]) - 5
        else:
            sunset_time = int(sunset_time[colon_loc-1]) - 4

        return sunset_time % 24

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
c.select_trail(name="Algonquin Provincial Park Canoe Routes")
c.extract_data()
# c.organize_events("Algonquin Provincial Park Canoe Routes")
#c.extract_topographical_data()

# w = Weather(1, 45.0, -79.0)
# w.score_each_hour()
