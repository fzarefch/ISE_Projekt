import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

# Verbindung zur PostgreSQL-Datenbank
engine = create_engine('postgresql://postgres:Rayan1388@localhost/restaurant_data')

# Funktion zum Laden der Daten und Extrahieren der Geokoordinaten
def load_violation_data():
    query = """
    SELECT
        substring(geocoded_location FROM '\\(([0-9\\.-]+),') AS latitude,
        substring(geocoded_location FROM ', ([0-9\\.-]+)\\)') AS longitude,
        restaurant_name,
        violation_description
    FROM restaurant_inspection_violations
    """
    df = pd.read_sql(query, con=engine)
    # Umwandlung der Spalten in Float
    df[['latitude', 'longitude']] = df[['latitude', 'longitude']].astype(float)
    return df.dropna()

# Streamlit App
st.set_page_config(page_title="Restaurant Violations Dashboard", layout="wide")

# Titel und Beschreibung
st.title("Restaurant Violations Dashboard")
st.markdown("""
Dieses Dashboard zeigt die Dichte von Restaurantverstößen basierend auf den Inspektionsdaten. 
Die Heatmap visualisiert die geographische Verteilung der Verstöße.
""")

# Daten laden
with st.spinner("Daten werden geladen..."):
    df = load_violation_data()

# Benutzersteuerung für den Zoom-Level
zoom_level = st.slider(
    "Zoom-Level der Karte",
    min_value=1,
    max_value=15,
    value=10,
    step=1,
    help="Ziehe den Schieberegler, um den Zoom-Level der Karte anzupassen."
)

# Heatmap erstellen
st.header("Violation Heatmap")
fig = px.density_mapbox(
    df,
    lat="latitude",
    lon="longitude",
    radius=5,  # Verkleinerter Radius für die Punkte
    mapbox_style="open-street-map",  # Kartenstil
    color_continuous_scale="Viridis",  # Farbskala
    title="Violation Density Heatmap",
    hover_data={
        "restaurant_name": True,  # Restaurantnamen im Popup anzeigen
        "violation_description": True,  # Verstöße im Popup anzeigen
        "latitude": False,
        "longitude": False
    }
)

# Automatisches Setzen des Zoom- und Sichtbereichs
fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(
            lat=df["latitude"].mean(),
            lon=df["longitude"].mean()
        ),
        zoom=zoom_level,  # Dynamischer Zoom-Level
    ),
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    dragmode="zoom",  # Interaktives Zoomen
    coloraxis_colorbar=dict(
        title="Dichte",
        thicknessmode="pixels",
        thickness=20,
        lenmode="pixels",
        len=300,
    )
)

# Heatmap anzeigen
st.plotly_chart(fig, use_container_width=True)

# Info-Hinweis für Nutzer
st.info("Nutze den Schieberegler, um den Zoom-Level der Karte anzupassen. Interaktives Zoomen mit der Maus wird unterstützt.")