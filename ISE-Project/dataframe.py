# dataframe.py

import pandas as pd


class Dataframe:

    def __init__(self, file: str):
        """
        Initialisiert den Dataframe
        :param file: Pfad zur CSV-Datei
        """
        self.file = file
        self.data = None

    def __getitem__(self, column):
        return self.data[column]

    def load_data(self):
        self.data = pd.read_csv(self.file, sep=",")

    def read_parquet(self):
        self.data = pd.read_parquet(self.file)