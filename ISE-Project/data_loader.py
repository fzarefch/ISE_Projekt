# data_loader.py

import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from typing import Optional

from dataframe import Dataframe


class DataLoader:
    """
    Verantwortlich für den Zugriff auf die SQLite-Datenbank und das Laden der
    kombinierten Restaurant-Inspektionsdaten aus der Tabelle Inspection_Data.
    Implementiert als Singleton.
    """
    _instance: Optional["DataLoader"] = None

    def __new__(cls, *args, **kwargs) -> "DataLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_url: str) -> None:
        if not hasattr(self, '_initialized'):
            self.engine = create_engine(db_url)
            self._initialized = True

    def load_data(self, config_loader) -> Dataframe:
        """
        Lädt und bereinigt die Daten aus der SQLite-Datenbank.
        Zuerst wird geprüft, ob eine vorverarbeitete Parquet-Datei existiert.
        Falls ja, wird diese eingelesen, andernfalls werden die Daten aus der DB in ein
        Dataframe geladen, transformiert und als Parquet gespeichert.
        """
        cache_file = config_loader.get_parquet_config()
        df = Dataframe(cache_file)

        if os.path.exists(cache_file):
            st.write("Lade Daten aus dem Parquet-Cache...")
            df.read_parquet()
        else:
            st.write("Lade Daten aus der SQLite-Datenbank...")
            query = """
            SELECT
                geocoded_location,
                name AS restaurant_name,
                violation_description,
                inspection_type,
                inspection_date,
                city,
                zipcode AS zip_code,
                violation_code
            FROM Inspection_Data
            """

            df.data = pd.read_sql(query, con=self.engine)

            def parse_loc(loc: str) -> tuple[float, float]:
                if not isinstance(loc, str):
                    return None, None
                loc = loc.strip()
                if loc.startswith("(") and loc.endswith(")"):
                    loc = loc[1:-1]
                parts = loc.split(',')
                if len(parts) != 2:
                    return None, None
                try:
                    return float(parts[0].strip()), float(parts[1].strip())
                except Exception:
                    return None, None

            df.data['parsed'] = df.data['geocoded_location'].apply(parse_loc)
            df.data[['latitude', 'longitude']] = pd.DataFrame(df.data['parsed'].tolist(), index=df.data.index)
            df.data['inspection_date'] = pd.to_datetime(df.data['inspection_date'])
            df.data.drop(columns=['parsed'])
            df.data.dropna(subset=['latitude', 'longitude'])

            # Entferne Zeilen, bei denen latitude und longitude 0 (oder nahezu 0) sind
            epsilon = 1e-6
            df.data = df.data[(df['latitude'].abs() > epsilon) & (df.data['longitude'].abs() > epsilon)]
            df.data.to_parquet(cache_file, index=False)
        return df.data
