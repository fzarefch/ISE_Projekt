# sqlite_database.py

import sqlite3


class Database:
    conn = None
    cursor = None

    def __init__(self, name):
        """
        Initialisiert den Kontextmanager
        :param name: Name der Datenbank
        """
        self.name = name

    def __enter__(self):
        """
        Öffnet die Verbindung zur Datenbank
        """
        self.conn = sqlite3.connect(self.name)
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        schließt die Verbindung zur Datenbank
        """
        self.conn.commit()
        self.conn.close()

    def create_table(self, query):
        """
        Erstellt die Tabelle Inspection Data gemäß der YAML-Konfiguration.
        :param query: SQL-query zum Erstellen der Tabelle
        """
        self.cursor.execute(query)
        self.conn.commit()

    def insert_data(self, data, query):
        """
        Fügt Daten aus einem DataFrame in die Tabelle Inspection_Data ein.
        :param data: DataFrame mit den Daten
        :param query: SQL-query zum Einfügen der Daten
        """
        for _, row in data.iterrows():
            self.cursor.execute(query, (
                row['name'], row['street_address'], row['city'], row['zipcode'], row['geocoded_location'],
                row['inspection_date'], row['inspection_type'], row['violation_code'], row['violation_description']
            ))
        self.conn.commit()
