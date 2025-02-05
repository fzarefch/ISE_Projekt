import pandas as pd
from config_loader import ConfigLoader


class CSVLoader:


    def __init__(self, csv_file_path: str):
        """
        Initialisiert den CSVLoader.
        :param csv_file_path: Pfad zur CSV-Datei
        """
        self._csv_file_path = csv_file_path

    def load(self) -> pd.DataFrame:
        """
        Lädt die CSV-Datei in ein Pandas DataFrame.
        :return: DataFrame mit den CSV-Daten
        """
        return pd.read_csv(self._csv_file_path)

