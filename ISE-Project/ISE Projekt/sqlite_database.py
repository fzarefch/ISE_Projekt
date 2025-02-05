import sqlite3


class Database:
    conn = None
    cursor = None

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.conn = sqlite3.connect(self.name)
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.commit()
        self.conn.close()
