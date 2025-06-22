import numpy as np
import sys
import io
import base64
import geopandas as gpd
from matplotlib import pyplot as plt
from google import genai

sys.path.insert(1, 'api/')
from main import Calculate, Weather

def plot_trail(trail_gdf):
    fig, ax = plt.subplots()
    trail_gdf.plot(ax=ax, color='black')
    ax.set_aspect('equal')
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

calculate = Calculate()
calculate.select_trail("Algonquin Provincial Park Canoe Routes")
#gdf = calculate.selected_trail
#gdf = plot_trail(gdf)

calculate.index_campsites([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.05, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
x,y,z = calculate.extract_data()
weather = Weather(1, 45.0, -79.0)
print(np.nansum(calculate.overlay_weather_over_veg_secondary(x, weather.score_each_hour(), y,z)))


# plt.figure(figsize=(10, 8))

# plt.pcolormesh(x_edges, y_edges, veg_density_grid, shading='auto', cmap='Greens')
# plt.colorbar(label="Vegetation Density")

# if calculate.geom_type == "LineString":
#     x, y = trail.xy
#     plt.plot(x, y, color='black', linewidth=2, label="Trail")
# else:
#     for line in trail.geoms:
#         x, y = line.xy
#         plt.plot(x, y, color='black', linewidth=2)

# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Trail and Vegetation Density")
# plt.legend()
# plt.axis("equal")
# plt.show()
