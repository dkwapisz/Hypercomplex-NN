import configparser
import os


class PropertiesResolver:
    def __init__(self, properties_file):
        self.config = configparser.ConfigParser()
        self.properties_file = properties_file
        if os.path.exists(properties_file):
            self.config.read(properties_file)
        else:
            print(f"Warning: Properties file '{properties_file}' not found. A new one will be created.")

    def get(self, section, key, default=None):
        return self.config.get(section, key, fallback=default)

    def display(self):
        for section in self.config.sections():
            print(f"[{section}]")
            for key, value in self.config.items(section):
                print(f"{key} = {value}")
        print()
