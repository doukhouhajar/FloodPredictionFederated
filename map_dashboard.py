import folium
import pandas as pd
from fusion_predict import predict_rainfall_lstm, predict_flood_risk_gnn
# Load station data
stations = pd.read_csv("GNN/station_region_map_climate.csv")

m = folium.Map(location=[-1.94, 29.87], zoom_start=5)

for i, row in stations.iterrows():
    rainfall = predict_rainfall_lstm(f"LSTM/data/{row['ID']}.csv")  # individual station files
    flood_risk = predict_flood_risk_gnn(i)

    alert_level = "Low"
    if rainfall > 0.5 and flood_risk > 0.5:
        alert_level = "High"
    elif rainfall > 0.3 or flood_risk > 0.3:
        alert_level = "Medium"

    popup = f"{row['ID']}<br>Rainfall: {rainfall:.2f}<br>Flood Risk: {flood_risk:.2f}<br>Alert: {alert_level}"
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=popup,
        icon=folium.Icon(color="red" if "High" in alert_level else "orange" if "Medium" in alert_level else "green")
    ).add_to(m)

m.save("flood_dashboard.html")