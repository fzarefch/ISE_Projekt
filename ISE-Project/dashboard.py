# dashboard.py

import streamlit as st

from config_loader import ConfigLoader
from data_loader import DataLoader
from views.heatmaps_view import HeatmapsView
from views.analysis_view import AnalysisView
from views.location_clustering_view import LocationClusteringView
from views.violation_clustering_view import ViolationClusteringView

# Setze die Seitenkonfiguration (falls nicht schon in den Modulen geschehen)
st.set_page_config(page_title="Restaurant Violations Dashboard", layout="wide")

# Erzeuge den ConfigLoader um die YAML Konfigurationen zu laden
yaml_path = "database_configuration.yaml"
config_loader = ConfigLoader(yaml_path)

# Erzeuge den DataLoader und lade die Daten
db_url = config_loader.get_database_config()["url"]
loader = DataLoader(db_url=db_url)
data = loader.load_data(config_loader)

# Erstelle die View-Instanzen
heatmap_view = HeatmapsView(data)
analysis_view = AnalysisView(data)
location_clustering_view = LocationClusteringView(data)
violation_clustering_view = ViolationClusteringView(data)

# Erstelle die Tabs und rendere die Views
tabs = st.tabs([
    "Heatmaps",
    "Verstöße-Analysen",
    "K-Means Location Clustering",
    "K-Means Violation Clustering"
])
with tabs[0]:
    heatmap_view.render()
with tabs[1]:
    analysis_view.render()
with tabs[2]:
    location_clustering_view.render()
with tabs[3]:
    violation_clustering_view.render()
