# data_transformer.py

import pandas as pd

from dataframe import Dataframe


class DataTransformer:
    def __init__(self, df: Dataframe):
        """
        Initialisiert den DataTransformer.
        :param df: DataFrame mit den zu transformierenden Daten
        """
        self.df = df

    def transform(self) -> Dataframe:
        """
        Transformiert die Daten:
        - Fehlende Werte für Zipcode prüfen und Zeilen entfernen
        - Fehlende Werte in der Geocoded Location-Spalte mit einem Standardwert füllen
        - Geocoded Location bereinigen
        - Spalten umbenennen
        - Datum ins richtige Format bringen
        :return: Transformierter DataFrame
        """
        # Fehlende Werte für Zipcode prüfen und Zeilen entfernen
        self.df.data = self.df.data.dropna(subset=['Food Establishment Zip Code'])

        # Fehlende Werte in der Geocoded Location-Spalte mit einem Standardwert füllen
        self.df.data['Geocoded Location'] = self.df.data['Geocoded Location'].fillna("(0.0, 0.0)")

        # Geocoded Location bereinigen
        self.df.data['geocoded_location'] = self.df.data['Geocoded Location'].apply(lambda x: x.split("\n")[-1].strip())

        # Spalten umbenennen
        self.df.data.rename(columns={
            'Food Establishment Name': 'name',
            'Food Establishment Street Address': 'street_address',
            'Food Establishment City': 'city',
            'Food Establishment Zip Code': 'zipcode',
            'Inspection Date': 'inspection_date',
            'Inspection Type': 'inspection_type',
            'Violation Code': 'violation_code',
            'Violation Description': 'violation_description'
        }, inplace=True)

        # Datum ins richtige Format umwandeln
        self.df.data['inspection_date'] = pd.to_datetime(self.df.data['inspection_date'], format='%m/%d/%Y').dt.date

        # Zipcode in Integer umwandeln
        self.df.data['zipcode'] = pd.to_numeric(self.df.data['zipcode'], errors="coerce").fillna(0).astype(int)

        # Nur die benötigten Spalten auswählen
        self.df.data = self.df.data[[
            'name', 'street_address', 'city', 'zipcode', 'geocoded_location',
            'inspection_date', 'inspection_type', 'violation_code', 'violation_description'
        ]]

        return self.df.data