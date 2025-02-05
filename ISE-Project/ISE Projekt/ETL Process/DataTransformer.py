import pandas as pd



class DataTransformer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialisiert den DataTransformer.
        :param data: DataFrame mit den zu transformierenden Daten
        """
        self._data = data.copy()  # Vermeidet SettingWithCopyWarning

    def transform(self) -> pd.DataFrame:
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
        self._data = self._data.dropna(subset=['Food Establishment Zip Code'])

        # Fehlende Werte in der Geocoded Location-Spalte mit einem Standardwert füllen
        self._data['Geocoded Location'] = self._data['Geocoded Location'].fillna("(0.0, 0.0)")

        # Geocoded Location bereinigen
        self._data['geocoded_location'] = self._data['Geocoded Location'].apply(lambda x: x.split("\n")[-1].strip())

        # Spalten umbenennen
        self._data.rename(columns={
            'Food Establishment Name': 'name',
            'Food Establishment Street Address': 'street_address',
            'Food Establishment City': 'city',
            'Food Establishment Zip Code': 'zipcode',
            'Inspection Date': 'inspection_date',
            'Inspection Type': 'inspection_type',
            'Violation Code': 'violation_code',
            'Violation Description': 'violation_description'
        }, inplace=True)

        # Datum ins richtige Format bringen
        self._data['inspection_date'] = pd.to_datetime(self._data['inspection_date'], format='%m/%d/%Y').dt.date

        # Zipcode in Integer umwandeln
        self._data['zipcode'] = pd.to_numeric(self._data['zipcode'], errors="coerce").fillna(0).astype(int)

        # Nur die benötigten Spalten auswählen
        self._data = self._data[[
            'name', 'street_address', 'city', 'zipcode', 'geocoded_location',
            'inspection_date', 'inspection_type', 'violation_code', 'violation_description'
        ]]

        return self._data