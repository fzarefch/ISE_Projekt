from datetime import datetime

import pandas as pd
from config_loader import ConfigLoader
from sqlite_database import Database
from dataframe import Dataframe

# Erzeuge den ConfigLoader um die YAML Konfigurationen zu laden
yaml_path = "database_configuration.yaml"
config_loader = ConfigLoader(yaml_path)


# CSV-Datei laden
csv_filename = config_loader.get_csv_config()["location"]
df = Dataframe(csv_filename)
df.load_data()


# Fehlende Werte für Zipcode prüfen und Zeilen entfernen
df.data = df.data.dropna(subset=['Food Establishment Zip Code'])

# Fehlende Werte in der Geocoded Location-Spalte mit einem Standardwert füllen
df.data['Geocoded Location'] = df.data['Geocoded Location'].fillna("(0.0, 0.0)")

# Transformation: Geocoded Location bereinigen
df.data['geocoded_location'] = df.data['Geocoded Location'].apply(lambda x: x.split("\n")[-1].strip())

# Spalten umbenennen
df.data.rename(columns={
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
df.data['inspection_date'] = pd.to_datetime(df.data['inspection_date'], format='%m/%d/%Y').dt.date

# SQLite-Datenbank-Datei
db_filename = config_loader.get_database_config()["location"]

with Database(db_filename) as db:

    # Tabellenstruktur erstellen
    create_combined_table = config_loader.get_queries()["create_table"]
    db.cursor.execute(create_combined_table)

    # Daten in die kombinierte Tabelle einfügen
    for _, row in df.data.iterrows():
        db.cursor.execute(config_loader.get_queries()["insert_data"], (
            row['name'], row['street_address'], row['city'], row['zipcode'], row['geocoded_location'],
            row['inspection_date'], row['inspection_type'], row['violation_code'], row['violation_description']
        ))

print("ETL-Prozess abgeschlossen. Daten wurden erfolgreich in die kombinierte Tabelle geladen.")