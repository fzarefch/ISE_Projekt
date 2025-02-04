# views/heatmaps_view.py

import streamlit as st
import plotly.express as px
import pandas as pd
from utils.decorators import log_execution, time_execution

class HeatmapsView:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    @log_execution
    @time_execution
    def render(self) -> None:
        st.title("Restaurant Violations Heatmaps")
        st.markdown("Dieses Dashboard zeigt die Dichte von Restaurantverstößen.")

        df = self.data
        selected_type = st.selectbox("Wähle den Inspektionstyp", options=df['inspection_type'].unique(), index=0)
        selected_violation = st.selectbox("Wähle eine Verstoßart", options=df['violation_description'].unique(), index=0)
        filtered_df = df[(df['inspection_type'] == selected_type) & (df['violation_description'] == selected_violation)]
        st.header("Violation Heatmap")
        fig = px.density_mapbox(
            filtered_df,
            lat="latitude",
            lon="longitude",
            radius=5,
            color_continuous_scale="Plasma",
            mapbox_style="open-street-map",
            title=f"Violation Density Heatmap: {selected_type}, {selected_violation}",
            hover_data={"restaurant_name": True, "violation_description": True}
        )
        if not filtered_df.empty:
            min_lat, max_lat = filtered_df["latitude"].min(), filtered_df["latitude"].max()
            min_lon, max_lon = filtered_df["longitude"].min(), filtered_df["longitude"].max()
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            fig.update_layout(
                mapbox=dict(
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=5
                )
            )
        else:
            center_lat = df["latitude"].mean()
            center_lon = df["longitude"].mean()
            fig.update_layout(
                mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=5)
            )
        fig.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            dragmode="zoom",
            coloraxis_colorbar=dict(
                title="Dichte",
                thicknessmode="pixels",
                thickness=20,
                lenmode="pixels",
                len=300,
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})