# views/location_clustering_view.py

import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.decorators import log_execution, time_execution

class LocationClusteringView:
    def __init__(self, data) -> None:
        self.data = data

    @log_execution
    @time_execution
    def render(self) -> None:
        st.title("K-Means Clustering nach Location (Geokoordinaten)")
        st.markdown("""
        **Ziel:** Restaurants nach ihrer geographischen Lage clustern, um Hotspots und regionale Schwerpunkte zu identifizieren.
        """)
        df = self.data
        k = st.slider("Anzahl der Cluster (k) für Location-Clustering", min_value=2, max_value=10, value=4, key='location_cluster_slider')
        location_data = df[['latitude', 'longitude']]
        scaler = StandardScaler()
        location_data_scaled = scaler.fit_transform(location_data)
        kmeans_loc = KMeans(n_clusters=k, random_state=42)
        kmeans_loc.fit(location_data_scaled)
        df['cluster_location'] = kmeans_loc.labels_.astype(str)
        cluster_centers = scaler.inverse_transform(kmeans_loc.cluster_centers_)
        center_lat = cluster_centers[:, 0].mean()
        center_lon = cluster_centers[:, 1].mean()
        fig = px.scatter_mapbox(
            df,
            lat='latitude',
            lon='longitude',
            color='cluster_location',
            mapbox_style='open-street-map',
            title=f"K-Means Clustering (k={k}) nach Geokoordinaten"
        )
        fig.update_layout(
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=5
            ),
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})