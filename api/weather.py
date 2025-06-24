import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY_TOMORROW")


class Weather:
    def __init__(self, date, lat, long, length, legs=None):
        self.url = "https://api.tomorrow.io/v4/timelines"
        self.date = date
        self.lat = lat
        self.long = long
        self.length = length

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
        final = []
        print(self.length / 24)
        for i in range(int(self.length / 24)):
            # Example: use today's date at midnight UTC
            start_time = datetime.datetime.now() + datetime.timedelta(days=i)
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
            params = {
                "location": f"{self.lat},{self.long}",
                "fields": [
                    "temperature", "precipitationIntensity", "windSpeed",
                    "humidity", "cloudCover", "dewPoint", "visibility", "weatherCode"
                ],
                "timesteps": "1h",
                "units": "metric",
                "startTime": start_time,
                "apikey": API_KEY
            }
            response = requests.get(self.url, params=params)
            data = response.json()
            print(data)
            hourly_score = []
            for i in range(24):
                if "code" in data and data["code"] == 429001:
                    print("rate_limit")
                    break  # rate limit

                if "data" not in data:
                    hourly_score.append({})
                    continue

                v = data["data"]["timelines"][0]["intervals"][i]["values"]
                hourly_score.append(self.score_hour(v))
            final += hourly_score

        return final

    def __repr__(self):
        return f"Weather(date={self.date}, temperature={self.temperature}, precipitation={self.precipitation}, wind_speed={self.wind_speed})"

    def find_sunrise(self):
        params = {
            "location": f"{self.lat},{self.long}",
            "fields": "sunriseTime",
            "timesteps": "1d",
            "units": "metric",
            "apikey": API_KEY
        }
        response = requests.get(self.url, params=params)
        data = response.json()

        sunrise_time = data["data"]["timelines"][0]["intervals"][0]["values"]["sunriseTime"]

        colon_loc = sunrise_time.index(":")
        if int(sunrise_time[colon_loc + 1] + sunrise_time[colon_loc + 2]) <= 30:
            sunrise_time = int(sunrise_time[colon_loc - 1]) - 5
        else:
            sunrise_time = int(sunrise_time[colon_loc - 1]) - 4

        return sunrise_time % 24

    def find_sunset(self):
        params = {
            "location": f"{self.lat},{self.long}",
            "fields": "sunsetTime",
            "timesteps": "1d",
            "units": "metric",
            "apikey": API_KEY
        }
        response = requests.get(self.url, params=params)
        data = response.json()

        sunset_time = data["data"]["timelines"][0]["intervals"][0]["values"]["sunsetTime"]
        colon_loc = sunset_time.index(":")
        if int(sunset_time[colon_loc + 1] + sunset_time[colon_loc + 2]) <= 30:
            sunset_time = int(sunset_time[colon_loc - 1]) - 5
        else:
            sunset_time = int(sunset_time[colon_loc - 1]) - 4

        return sunset_time % 24
