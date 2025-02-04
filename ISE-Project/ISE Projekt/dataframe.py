import pandas as pd


class Dataframe:
    file = None
    data = None

    def __init__(self, file):
        self.file = file

    def __getitem__(self, column):
        return self.data[column]

    def load_data(self):
        self.data = pd.read_csv(self.file, sep=",")
