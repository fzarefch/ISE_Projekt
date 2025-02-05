import sqlite3
import pandas as pd


class DatabaseConnector:
    def __init__(self, database_name: str, create_table_query: str, insert_data_query: str) -> None:
        """
        Initialisiert den Kontextmanager.
        :param database_name: Name der Datenbank
        :param create_table_query: SQL-Abfrage zum Erstellen der Tabelle
        :param insert_data_query: SQL-Abfrage zum Einfügen von Daten
        """
        self._database_name = database_name
        self._create_table_query = create_table_query
        self._insert_data_query = insert_data_query

    def __enter__(self):
        """
        Öffnet die Verbindung zur Datenbank.
        :return: DatabaseConnector
        """
        self._connection = sqlite3.connect(self._database_name)
        self._cursor = self._connection.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Schließt die Verbindung zur Datenbank.
        """
        self._connection.close()

    def create_table(self):
        """
        Erstellt die Tabelle Inspection_Data gemäß der YAML-Konfiguration.
        """
        self._cursor.execute(self._create_table_query)
        self._connection.commit()

    def insert_data(self, data: pd.DataFrame):
        """
        Fügt Daten aus einem DataFrame in die Tabelle Inspection_Data ein.
        :param data: DataFrame mit den Daten
        """
        # Konvertiere den DataFrame in eine Liste von Tupeln
        data_tuples = list(data.itertuples(index=False, name=None))

        # Führe die SQL-Abfrage aus
        self._cursor.executemany(self._insert_data_query, data_tuples)
        self._connection.commit()

    def fetch_all(self):
        """
        Holt alle Daten aus der Tabelle Inspection_Data.
        :return: Alle Zeilen als Liste von Tupeln
        """
        result = self._cursor.execute("SELECT * FROM Inspection_Data")
        return result.fetchall()