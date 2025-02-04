import pandas as pd


class Dataframe:
    data = None

    def __init__(self, file):
        self.file = file

    def __getitem__(self, column):
        return self.data[column]

    def load_data(self):
        self.data = pd.read_csv(self.file, sep=",")

    def read_parquet(self):
        self.data = pd.read_parquet(self.file)

    def read_sql(self, query, engine):
        self.data = pd.read_sql(query, con=engine)