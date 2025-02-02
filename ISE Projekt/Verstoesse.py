import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import altair as alt

# Streamlit-Konfiguration
st.set_page_config(page_title="Restaurant Violations Dashboard", layout="wide")

# Verbindung zur PostgreSQL-Datenbank
engine = create_engine('postgresql://postgres:Rayan1388@localhost/restaurant_data')

# Funktion zum Laden der Daten
def load_data():
    query = """
    SELECT
        substring(geocoded_location FROM '\\(([0-9\\.-]+),') AS latitude,
        substring(geocoded_location FROM ', ([0-9\\.-]+)\\)') AS longitude,
        violation_description, restaurant_name
    FROM restaurant_inspection_violations
    """
    try:
        df = pd.read_sql(query, con=engine)
        df[['latitude', 'longitude']] = df[['latitude', 'longitude']].astype(float)
        df = df.dropna(subset=['latitude', 'longitude', 'violation_description'])
        return df
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        st.stop()

# Daten laden
df = load_data()

# Dashboard-Überschrift
st.title("Restaurant Violations Analysis")
st.markdown("""
Dieses Dashboard zeigt:
- Die **Top 10** häufigsten Verstöße.
- Die **Top 10** häufigsten Kombinationen von Verstößen.
""")

# Häufigkeitsanalyse der Verstöße
st.header("Top 10 häufigste Verstöße")
if df['violation_description'].empty:
    st.warning("Es sind keine Verstöße vorhanden.")
else:
    violation_counts = df['violation_description'].value_counts().head(10).reset_index()
    violation_counts.columns = ['Violation', 'Frequency']

    # Altair-Chart als vertikales Balkendiagramm
    chart = alt.Chart(violation_counts).mark_bar().encode(
        y=alt.Y('Violation:N', sort='-x', title='Verstoß'),  # Namen auf der Y-Achse
        x=alt.X('Frequency:Q', title='Häufigkeit'),
        tooltip=['Violation', 'Frequency']
    ).properties(
        width=800,
        height=400  # Mehr Platz für die Namen
    )
    st.altair_chart(chart)

# Häufigste Kombinationen von Verstößen
st.header("Häufigste Kombinationen von Verstößen")
try:
    # Verstöße in ein One-Hot-Encoded-Format umwandeln
    df_association = df['violation_description'].str.get_dummies(sep=', ')

    # Kombinationshäufigkeiten berechnen
    combo_matrix = df_association.T.dot(df_association)

    # Nur obere Dreiecksmatrix extrahieren, um Doppelte zu vermeiden
    combo_matrix = combo_matrix.where(np.triu(np.ones(combo_matrix.shape), k=1).astype(bool))

    # Kombinationen in ein DataFrame umwandeln
    combo_counts = combo_matrix.stack().reset_index()
    combo_counts.columns = ['Violation 1', 'Violation 2', 'Frequency']

    # Nur Kombinationen mit Frequenz > 0 und Top 10 auswählen
    combo_counts = combo_counts[combo_counts['Frequency'] > 0]
    combo_counts = combo_counts.sort_values(by='Frequency', ascending=False).head(10)

    # Kombinationen für Visualisierung zusammenführen
    combo_counts['Combination'] = combo_counts['Violation 1'] + " & " + combo_counts['Violation 2']

    # Altair-Chart als vertikales Balkendiagramm
    combo_chart = alt.Chart(combo_counts).mark_bar().encode(
        y=alt.Y('Combination:N', sort='-x', title='Verstoßkombination'),  # Kombinationen auf der Y-Achse
        x=alt.X('Frequency:Q', title='Häufigkeit'),
        tooltip=['Combination', 'Frequency']
    ).properties(
        width=800,
        height=400  # Mehr Platz für die Kombinationen
    )
    st.altair_chart(combo_chart)

except Exception as e:
    st.error(f"Fehler bei der Analyse der Kombinationen: {e}")