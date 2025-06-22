import numpy as np
from api.main import Calculate, Weather

calculate = Calculate()
calculate.select_trail("Algonquin Provincial Park Canoe Routes")
calculate.index_campsites([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.05, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
x,y,z = calculate.extract_data()
weather = Weather(1, 45.0, -79.0)
print(np.nansum(calculate.overlay_weather_over_veg_secondary(x, weather.score_each_hour(), y,z)))