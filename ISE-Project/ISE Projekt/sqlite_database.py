import sqlite3

class Database:
    db = None
    cursor = None

    def __init__(self, name):
        self.db = sqlite3.connect(name)
        self.cursor = self.db.cursor()

    def close(self):
        self.db.commit()
        self.db.close()

    def save_changes(self):
        self.db.commit()