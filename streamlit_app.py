import streamlit as st
import pandas as pd
import requests
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import pandas as pd
import math
from pathlib import Path
from datetime import datetime, timedelta
import io
import pydeck as pdk
import numpy as np

# -----------------------------------------------------------------------------

st.title("USGS Geomagnetic Data - BOU Station")

# Constants
STATIONS = {
    "BOU": {"lat": 40.137, "lon": -105.237},
    "BDT": {"lat": 46.283, "lon": -96.617}
}
BOU_COORDINATES = {"lat": 40.137, "lon": -105.237}
BASE_URL = "https://geomag.usgs.gov/ws/data/"

# Time range: last 24 hours
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=24)

# Format times
start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")

params = {
    "id": "BOU",
    "format": "iaga2002",
    "elements": "X,Y,Z,F",
    "starttime": start_str,
    "endtime": end_str
}

st.markdown(f"### Fetching data from {start_str} to {end_str} UTC")

# Fetch data
response = requests.get(BASE_URL, params=params)

if response.status_code == 200:
    raw_data = response.text

    # Parse IAGA2002 format (skip header, read columns)
    lines = raw_data.splitlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("DATE"):
            data_start = i
            break

    # Read data into DataFrame
    data_io = io.StringIO("\n".join(lines[data_start:]))
    df = pd.read_csv(data_io, delim_whitespace=True, parse_dates=[[0, 1]], na_values=["99999.00", "99999.000"])
    df.columns = ['Date','time', 'Y','X', 'Y', 'Z', 'F']

    # Drop missing values
    df.dropna(inplace=True)

    # Display map
    st.map(pd.DataFrame([BOU_COORDINATES], columns=["lat", "lon"]))

    # Plot time series
    st.line_chart(df.set_index("Date")[["time",'Y','X', 'Y', 'Z', 'F']])
else:
    st.error(f"Failed to retrieve data. Status code: {response.status_code}")

# ---
# st.title("USGS Geomagnetic Data - BOU & BDT Stations")

# Station coordinates
STATIONS = {
    "BOU": {"lat": 40.137, "lon": -105.237},
    "BRW": {"lat": 46.283, "lon": -96.617}
}

BASE_URL = "https://geomag.usgs.gov/ws/data/?id="
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=24)
start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")

params_template = {
    "format": "iaga2002",
    "elements": "X,Y,Z,F",
    "starttime": start_str,
    "endtime": end_str
}    
@st.cache_data(ttl=3600)
def fetch_station_data(station_id):
    params = params_template.copy()
    params["id"] = station_id
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        st.error(f"Failed to retrieve data for {station_id}")
        return None

    lines = response.text.splitlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("DATE"):
            data_start = i
            break
    data_io = io.StringIO("\n".join(lines[data_start:]))
    df = pd.read_csv(data_io, delim_whitespace=True, parse_dates=[[0, 1]], na_values=["99999.00", "99999.000"])
    df.columns = ['Date','time','dY', 'X', 'Y', 'Z', 'F']
    df.dropna(inplace=True)
    return df
 # Fetch and display both stations
dataframes = {}
for station in STATIONS:
    df = fetch_station_data(station)
    if df is not None:
        st.subheader(f"{station} Magnetic Field Components")
        st.line_chart(df.set_index("Date")[['time','dY','X', 'Y', 'Z', 'F']])
        dataframes[station] = df

# Prepare heatmap data from the last reading of each station
heatmap_data = []
for station, df in dataframes.items():
    if not df.empty:
        latest = df.iloc[-1]
        heatmap_data.append({
            "lat": STATIONS[station]["lat"],
            "lon": STATIONS[station]["lon"],
            "F": latest["F"]
        })

# Create DataFrame for heatmap
heatmap_df = pd.DataFrame(heatmap_data)

if not heatmap_df.empty:
    st.subheader("Magnetic Field Intensity Heatmap (F)")
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=43,
            longitude=-101,
            zoom=4,
            pitch=40,
        ),
        layers=[
            pdk.Layer(
                "HeatmapLayer",
                data=heatmap_df,
                get_position='[lon, lat]',
                get_weight="F",
                radiusPixels=60,
                aggregation='MEAN'
            )
        ]
    ))
else:
    st.warning("No data available to generate heatmap.")   
# -----------------------------------------------------------------------------
# Draw the actual page

# Title
st.title("Open Source Map of America")

# Create a Folium map centered on the geographic center of the USA
us_center_coords = [39.8283, -98.5795]  # Approximate center of the continental USA

# Create the map
m = folium.Map(location=us_center_coords, zoom_start=4, tiles='OpenStreetMap')

# Optional: Add marker for Washington D.C.
folium.Marker(
    location=[38.9072, -77.0369],
    popup='Washington, D.C.',
    icon=folium.Icon(color='blue')
).add_to(m)

# Display map
folium_static(m)

# Set the title that appears at the top of the page.


