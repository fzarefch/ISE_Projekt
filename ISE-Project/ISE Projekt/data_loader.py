# data_loader.py

import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from typing import Optional


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

    def load_data(self) -> pd.DataFrame:
        """
        Lädt und bereinigt die Daten aus der SQLite-Datenbank.
        Zuerst wird geprüft, ob eine vorverarbeitete Parquet-Datei existiert.
        Falls ja, wird diese eingelesen, andernfalls werden die Daten aus der DB
        geladen, transformiert und als Parquet gespeichert.
        """
        cache_file = "cached_inspection_data.parquet"
        if os.path.exists(cache_file):
            st.write("Lade Daten aus dem Parquet-Cache...")
            df = pd.read_parquet(cache_file)
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
            df = pd.read_sql(query, con=self.engine)

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

            df['parsed'] = df['geocoded_location'].apply(parse_loc)
            df[['latitude', 'longitude']] = pd.DataFrame(df['parsed'].tolist(), index=df.index)
            df['inspection_date'] = pd.to_datetime(df['inspection_date'])
            df = df.drop(columns=['parsed'])
            df = df.dropna(subset=['latitude', 'longitude'])
            # Entferne Zeilen, bei denen latitude und longitude 0 (oder nahezu 0) sind
            epsilon = 1e-6
            df = df[(df['latitude'].abs() > epsilon) & (df['longitude'].abs() > epsilon)]
            df.to_parquet(cache_file, index=False)
        return df
