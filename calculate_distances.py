import pandas as pd
from math import radians, sin, cos, sqrt, atan2

df = pd.read_csv("data/port_master.csv")

origins = df[df["Port_Type"] == "Origin"]
destinations = df[df["Port_Type"] == "Destination"]

def haversine_nm(lat1, lon1, lat2, lon2):
    """Real great-circle distance in nautical miles between two coordinates."""
    R_km = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance_km = R_km * c
    distance_nm = distance_km * 0.539957  # km to nautical miles
    return round(distance_nm, 1)

routes = []
for _, o in origins.iterrows():
    for _, d in destinations.iterrows():
        dist = haversine_nm(o["Latitude"], o["Longitude"], d["Latitude"], d["Longitude"])
        routes.append({
            "Origin_Port": o["Port_Name"],
            "Origin_Country": o["Country"],
            "Destination_Port": d["Port_Name"],
            "Distance_NM": dist
        })

route_df = pd.DataFrame(routes)
print(route_df.to_string(index=False))

route_df.to_csv("data/route_data.csv", index=False)
print("\nSaved to data/route_data.csv")