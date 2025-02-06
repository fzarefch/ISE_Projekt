# views/analysis_view.py

import streamlit as st
import plotly.express as px
import pandas as pd
from utils.decorators import log_execution, time_execution
import numpy as np
class AnalysisView:
    def __init__(self, data) -> None:
        self.data = data

    @log_execution
    @time_execution
    def render(self) -> None:
        st.title("Restaurant Violations Analysis")
        st.markdown("""
        In diesem Bereich findest du verschiedene Analysen:
        - KPIs (Key Performance Indicators)
        - Top häufigste Verstöße
        - Häufigste Kombinationen von Verstößen
        - Trends über Zeit
        - Verstöße nach Standort (Stadt & Postleitzahl)
        - Top Restaurants mit den meisten Verstößen
        """)
        df = self.data
        with st.expander("Filter für Städte, Postleitzahlen und Zeitraum anzeigen", expanded=False):
            cities = st.multiselect("Wähle Städte aus", options=df['city'].unique(), default=df['city'].unique())
            zip_codes = st.multiselect("Wähle Postleitzahlen aus", options=df['zip_code'].unique(), default=df['zip_code'].unique())
            min_date = df['inspection_date'].min()
            max_date = df['inspection_date'].max()
            date_range = st.date_input("Wähle einen Zeitraum", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, (list, tuple)) and len(date_range) >= 2:
            start_date = pd.Timestamp(date_range[0])
            end_date = pd.Timestamp(date_range[1])
        else:
            start_date = pd.Timestamp(date_range[0] if isinstance(date_range, (list, tuple)) else date_range)
            end_date = start_date
        filtered_data = df[(df['city'].isin(cities)) & (df['zip_code'].isin(zip_codes)) &
                           (df['inspection_date'].between(start_date, end_date))]
        if filtered_data.empty:
            st.warning("Keine Daten für die aktuellen Filter vorhanden.")
            st.stop()
        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5, sub_tab6 = st.tabs([
            "KPIs", "Top Verstöße", "Kombinationen", "Trends", "Standorte", "Top Restaurants"
        ])
        self.render_kpis(filtered_data, sub_tab1)
        self.render_top_violations(filtered_data, sub_tab2)
        self.render_combinations(filtered_data, sub_tab3)
        self.render_trends(filtered_data, sub_tab4)
        self.render_locations(filtered_data, sub_tab5)
        self.render_top_restaurants(filtered_data, sub_tab6)

    def render_kpis(self, df: pd.DataFrame, container: st.delta_generator.DeltaGenerator) -> None:
        with container:
            st.subheader("KPIs")
            total_inspections = df['inspection_date'].nunique()
            total_violations = len(df)
            avg_violations = total_inspections and total_violations / total_inspections or 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Inspektionen", total_inspections)
            col2.metric("Verstöße", total_violations)
            col3.metric("Ø Verstöße/Inspektion", round(avg_violations, 2))

    def render_top_violations(self, df: pd.DataFrame, container: st.delta_generator.DeltaGenerator) -> None:
        with container:
            st.subheader("Top häufigste Verstöße")
            top_n = st.slider("Wie viele Verstöße anzeigen?", min_value=5, max_value=30, value=10, step=1, key='top_violations_slider')
            counts = df['violation_description'].value_counts().head(top_n).reset_index()
            counts.columns = ['Violation', 'Frequency']
            fig = px.bar(counts, x='Frequency', y='Violation', orientation='h',
                         title=f"Top {top_n} Verstöße",
                         labels={'Frequency': 'Häufigkeit', 'Violation': 'Verstoß'},
                         color='Frequency', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

    def render_combinations(self, df: pd.DataFrame, container: st.delta_generator.DeltaGenerator) -> None:
        with container:
            st.subheader("Häufigste Kombinationen von Verstößen")
            top_n_combos = st.slider("Wie viele Kombinationen anzeigen?", min_value=5, max_value=30, value=10, step=1, key='combo_slider')
            try:
                assoc = df['violation_description'].str.get_dummies(sep=', ')
                combo_matrix = assoc.T.dot(assoc)
                combo_matrix = combo_matrix.where(np.triu(np.ones(combo_matrix.shape), k=1).astype(bool))
                combos = combo_matrix.stack().reset_index()
                combos.columns = ['Violation 1', 'Violation 2', 'Frequency']
                combos = combos[combos['Frequency'] > 0].sort_values(by='Frequency', ascending=False).head(top_n_combos)
                combos['Combination'] = combos['Violation 1'] + " & " + combos['Violation 2']
                fig = px.bar(combos, x='Frequency', y='Combination', orientation='h',
                             title=f"Top {top_n_combos} Kombinationen",
                             labels={'Frequency': 'Häufigkeit', 'Combination': 'Kombination'},
                             color='Frequency', color_continuous_scale='Plasma')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Fehler bei Kombinationen: {e}")

    def render_trends(self, df: pd.DataFrame, container: st.delta_generator.DeltaGenerator) -> None:
        with container:
            st.subheader("Trends über Zeit")
            df['year_month'] = df['inspection_date'].dt.to_period('M').astype(str)
            trends = df.groupby('year_month').size().reset_index(name='Frequency')
            fig = px.line(trends, x='year_month', y='Frequency',
                          title="Verstöße pro Monat", labels={'year_month': 'Monat', 'Frequency': 'Häufigkeit'},
                          markers=True)
            st.plotly_chart(fig, use_container_width=True)

    def render_locations(self, df: pd.DataFrame, container: st.delta_generator.DeltaGenerator) -> None:
        with container:
            top_n_city = st.slider("Anzahl Top Städte", min_value=5, max_value=30, value=10, key='city_slider_standorte')
            top_n_zip = st.slider("Anzahl Top Postleitzahlen", min_value=5, max_value=30, value=10, key='zip_slider_standorte')
            cities_data = df.groupby('city').size().reset_index(name='Frequency').sort_values(by='Frequency', ascending=False).head(top_n_city)
            zips_data = df.groupby('zip_code').size().reset_index(name='Frequency').sort_values(by='Frequency', ascending=False).head(top_n_zip)
            zips_data['zip_code'] = zips_data['zip_code'].astype(str)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(cities_data, x='Frequency', y='city', orientation='h',
                             title=f"Top {top_n_city} Städte mit den meisten Verstößen",
                             labels={'city': 'Stadt', 'Frequency': 'Häufigkeit'},
                             color='Frequency', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(zips_data, x='Frequency', y='zip_code', orientation='h',
                             title=f"Top {top_n_zip} Postleitzahlen mit den meisten Verstößen",
                             labels={'zip_code': 'Postleitzahl', 'Frequency': 'Häufigkeit'},
                             color='Frequency', color_continuous_scale='Blues')
                fig.update_yaxes(type='category')
                fig.update_traces(marker_line_width=2)
                st.plotly_chart(fig, use_container_width=True)

    def render_top_restaurants(self, df: pd.DataFrame, container: st.delta_generator.DeltaGenerator) -> None:
        with container:
            st.subheader("Top Restaurants")
            top_n_restaurants = st.slider("Wie viele Restaurants anzeigen?", min_value=5, max_value=30, value=10, key='restaurant_slider')
            restaurants_data = df.groupby('restaurant_name').size().reset_index(name='Frequency').sort_values(by='Frequency', ascending=False).head(top_n_restaurants)
            fig = px.bar(restaurants_data, x='Frequency', y='restaurant_name', orientation='h',
                         title=f"Top {top_n_restaurants} Restaurants mit den meisten Verstößen",
                         labels={'Frequency': 'Häufigkeit', 'restaurant_name': 'Restaurant'},
                         color='Frequency', color_continuous_scale='Oranges')
            st.plotly_chart(fig, use_container_width=True)