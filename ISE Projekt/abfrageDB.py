import pandas as pd
from sqlalchemy import create_engine

# Verbindung zur PostgreSQL-Datenbank herstellen
engine = create_engine('postgresql://postgres:Rayan1388@localhost/restaurant_data')

# SQL-Query zum Abrufen der Spaltennamen
query = """
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'restaurant_inspection_violations';
"""

# Abfrage ausführen
columns_df = pd.read_sql(query, con=engine)

# Spaltennamen ausgeben
print("Spalten in der Tabelle 'restaurant_inspection_violations':")
print(columns_df['column_name'].tolist())