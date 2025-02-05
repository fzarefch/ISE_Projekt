from DatabaseConnector import DatabaseConnector
from CSVLoader import CSVLoader
from DataTransformer import DataTransformer
from config_loader import ConfigLoader

if __name__ == "__main__":

    # YAML-Konfiguration laden
    #jeweils Pfad ändern!!!
    yaml_path = "C:/Users/loeck/Documents/Industrielle Softwareentwicklung/ISE Projekt/database_configuration.yaml"
    config_loader = ConfigLoader(yaml_path)

    # Datenbankkonfiguration laden
    database_config = config_loader.get_database_config()
    database_name = database_config.get("location")

    # CSV-Konfiguration laden
    csv_config = config_loader.get_csv_config()
    csv_file_path = csv_config.get("location")

    # SQL-Abfragen laden
    queries = config_loader.get_queries()
    create_table_query = queries.get("create_table")
    insert_data_query = queries.get("insert_data")

    # CSV-Datei laden
    csv_loader = CSVLoader(csv_file_path)
    data = csv_loader.load()

    # Daten transformieren
    transformer = DataTransformer(data)
    transformed_data = transformer.transform()

    # Daten in die Datenbank einfügen
    with DatabaseConnector(database_name, create_table_query, insert_data_query) as db:
        # Tabelle erstellen
        db.create_table()

        # Transformierte Daten in die Datenbank einfügen
        db.insert_data(transformed_data)

        # Daten aus der Datenbank abrufen und ausgeben
        result = db.fetch_all()
        for row in result:
            print(row)