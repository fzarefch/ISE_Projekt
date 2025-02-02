from sqlalchemy import create_engine
import pandas as pd

# Verbindung zur PostgreSQL-Datenbank mit SQLAlchemy
def connect_to_db():
    try:
        engine = create_engine(
            "postgresql+psycopg2://postgres:dein_passwort@localhost/restaurant_data"
        )
        return engine
    except Exception as e:
        print("Fehler bei der Verbindung:", e)
        return None

# Daten abfragen
def fetch_data(query):
    engine = connect_to_db()
    if engine:
        try:
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            print("Fehler bei der Abfrage:", e)
        finally:
            engine.dispose()