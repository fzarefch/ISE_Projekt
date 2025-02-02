import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine

# Verbindung zur PostgreSQL-Datenbank
engine = create_engine('postgresql://postgres:Rayan1388@localhost/restaurant_data')

# Funktion zum Laden der Daten für die Verstöße
@st.cache_data
def load_violation_data_cached():
    query = """
    SELECT
        substring(geocoded_location FROM '\\(([0-9\\.-]+),') AS latitude,
        substring(geocoded_location FROM ', ([0-9\\.-]+)\\)') AS longitude,
        restaurant_name,
        violation_description,
        inspection_type,
        inspection_date,
        city,
        zip_code
    FROM restaurant_inspection_violations
    """
    df = pd.read_sql(query, con=engine)
    df[['latitude', 'longitude']] = df[['latitude', 'longitude']].astype(float)
    df['inspection_date'] = pd.to_datetime(df['inspection_date'])
    return df.dropna()

# Streamlit App
st.set_page_config(page_title="Restaurant Violations Dashboard", layout="wide")

# Tabs für die beiden Funktionen
tab1, tab2 = st.tabs(["Heatmaps", "Verstöße-Analysen"])

# Tab 1: Heatmaps
with tab1:
    st.title("Restaurant Violations Heatmaps")
    st.markdown("""
    Dieses Dashboard zeigt die Dichte von Restaurantverstößen und erlaubt die Filterung nach Verstößen und Inspektionstypen.
    """)

    # Daten laden
    with st.spinner("Daten werden geladen..."):
        df = load_violation_data_cached()

    # Filter nach Inspektionstyp
    inspection_types = df['inspection_type'].unique()
    selected_type = st.selectbox("Wähle den Inspektionstyp", options=inspection_types, index=0)

    # Filter nach Verstoßart
    violation_types = df['violation_description'].unique()
    selected_violation = st.selectbox("Wähle eine Verstoßart", options=violation_types, index=0)

    # Filtere die Daten basierend auf der Auswahl
    filtered_df = df[(df['inspection_type'] == selected_type) & (df['violation_description'] == selected_violation)]

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
        filtered_df,
        lat="latitude",
        lon="longitude",
        radius=5,
        mapbox_style="open-street-map",
        color_continuous_scale="Viridis",
        title=f"Violation Density Heatmap: {selected_type}, {selected_violation}",
        hover_data={
            "restaurant_name": True,
            "violation_description": True,
            "latitude": False,
            "longitude": False
        }
    )

    # Automatisches Setzen des Zoom- und Sichtbereichs
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=filtered_df["latitude"].mean(),
                lon=filtered_df["longitude"].mean()
            ),
            zoom=zoom_level,
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        dragmode="zoom",
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

# Tab 2: Verstöße-Analysen
with tab2:
    st.title("Restaurant Violations Analysis")
    st.markdown("""
    Dieses Dashboard zeigt:
    - **Key Performance Indicators (KPIs)**
    - **Top 10 häufigste Verstöße**
    - **Top 10 häufigste Kombinationen von Verstößen**
    - **Trends über Zeit**
    - **Verstöße nach Stadt oder Postleitzahl**
    - **Top 10 Restaurants mit den meisten Verstößen**
    """)

    # Daten laden
    with st.spinner("Daten werden geladen..."):
        df = load_violation_data_cached()

    # Kompakte Filter für Städte, Postleitzahlen und Zeitraum
    with st.expander("Filter für Städte, Postleitzahlen und Zeitraum anzeigen"):
        cities = st.multiselect("Wähle Städte aus", options=df['city'].unique(), default=df['city'].unique())
        zip_codes = st.multiselect("Wähle Postleitzahlen aus", options=df['zip_code'].unique(), default=df['zip_code'].unique())
        date_range = st.date_input("Wähle einen Zeitraum", [df['inspection_date'].min(), df['inspection_date'].max()])

    # Konvertiere `date_range` zu `datetime64[ns]`
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    # Daten filtern
    filtered_data = df[
        (df['city'].isin(cities)) &
        (df['zip_code'].isin(zip_codes)) &
        (df['inspection_date'].between(start_date, end_date))
    ]

    # Key Performance Indicators
    st.header("Key Performance Indicators (KPIs)")
    total_inspections = filtered_data['inspection_date'].nunique()
    total_violations = len(filtered_data)
    average_violations_per_inspection = total_violations / total_inspections

    col1, col2, col3 = st.columns(3)
    col1.metric("Anzahl der Inspektionen", total_inspections)
    col2.metric("Gesamtzahl der Verstöße", total_violations)
    col3.metric("Durchschnittliche Verstöße pro Inspektion", round(average_violations_per_inspection, 2))

    # Top 10 häufigste Verstöße
    st.header("Top 10 häufigste Verstöße")
    violation_counts = filtered_data['violation_description'].value_counts().head(10).reset_index()
    violation_counts.columns = ['Violation', 'Frequency']

    fig = px.bar(
        violation_counts,
        x='Frequency',
        y='Violation',
        orientation='h',
        title="Top 10 häufigste Verstöße",
        labels={'Frequency': 'Häufigkeit', 'Violation': 'Verstoß'},
        color='Frequency',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # Häufigste Kombinationen von Verstößen
    st.header("Top 10 häufigste Kombinationen von Verstößen")
    try:
        df_association = filtered_data['violation_description'].str.get_dummies(sep=', ')
        combo_matrix = df_association.T.dot(df_association)
        combo_matrix = combo_matrix.where(np.triu(np.ones(combo_matrix.shape), k=1).astype(bool))
        combo_counts = combo_matrix.stack().reset_index()
        combo_counts.columns = ['Violation 1', 'Violation 2', 'Frequency']
        combo_counts = combo_counts[combo_counts['Frequency'] > 0].sort_values(by='Frequency', ascending=False).head(10)
        combo_counts['Combination'] = combo_counts['Violation 1'] + " & " + combo_counts['Violation 2']

        fig_combo = px.bar(
            combo_counts,
            x='Frequency',
            y='Combination',
            orientation='h',
            title="Top 10 häufigste Kombinationen von Verstößen",
            labels={'Frequency': 'Häufigkeit', 'Combination': 'Kombination'},
            color='Frequency',
            color_continuous_scale='Plasma'
        )
        fig_combo.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_combo, use_container_width=True)
    except Exception as e:
        st.error(f"Fehler bei der Analyse der Kombinationen: {e}")

    # Trends über Zeit
    st.header("Trends über Zeit")
    filtered_data['year_month'] = filtered_data['inspection_date'].dt.to_period('M').astype(str)
    violations_per_month = filtered_data.groupby('year_month').size().reset_index(name='Frequency')

    line_chart = px.line(
        violations_per_month,
        x='year_month',
        y='Frequency',
        title="Trends über Zeit: Anzahl der Verstöße pro Monat",
        labels={'year_month': 'Monat', 'Frequency': 'Häufigkeit'}
    )
    st.plotly_chart(line_chart, use_container_width=True)

    # Verstöße nach Stadt oder Postleitzahl
    st.header("Verstöße nach Stadt oder Postleitzahl")
    city_violations = filtered_data.groupby('city').size().reset_index(name='Frequency').sort_values(by='Frequency', ascending=False).head(10)
    zip_violations = filtered_data.groupby('zip_code').size().reset_index(name='Frequency').sort_values(by='Frequency', ascending=False).head(10)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Städte mit den meisten Verstößen")
        fig_city = px.bar(
            city_violations,
            x='Frequency',
            y='city',
            orientation='h',
            title="Verstöße nach Stadt",
            labels={'Frequency': 'Häufigkeit', 'city': 'Stadt'},
            color='Frequency',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_city, use_container_width=True)

    with col2:
        st.subheader("Top 10 Postleitzahlen mit den meisten Verstößen")
        fig_zip = px.bar(
            zip_violations,
            x='Frequency',
            y='zip_code',
            orientation='h',
            title="Verstöße nach Postleitzahl",
            labels={'Frequency': 'Häufigkeit', 'zip_code': 'Postleitzahl'},
            color='Frequency',
            color_continuous_scale='Blues'
        )
        fig_zip.update_layout(
            yaxis=dict(tickmode="linear", title="Postleitzahl"),
            xaxis=dict(title="Häufigkeit")
        )
        st.plotly_chart(fig_zip, use_container_width=True)

    # Top-Restaurants mit den meisten Verstößen
    st.header("Top 10 Restaurants mit den meisten Verstößen")
    top_restaurants = filtered_data.groupby('restaurant_name').size().reset_index(name='Frequency').sort_values(by='Frequency', ascending=False).head(10)

    fig_restaurant = px.bar(
        top_restaurants,
        x='Frequency',
        y='restaurant_name',
        orientation='h',
        title="Top 10 Restaurants mit den meisten Verstößen",
        labels={'Frequency': 'Häufigkeit', 'restaurant_name': 'Restaurant'},
        color='Frequency',
        color_continuous_scale='Oranges'
    )
    st.plotly_chart(fig_restaurant, use_container_width=True)