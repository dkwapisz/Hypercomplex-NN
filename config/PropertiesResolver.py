import json
import os


class PropertiesResolver:
    def __init__(self, properties_file):
        self.properties_file = properties_file
        if os.path.exists(properties_file):
            with open(properties_file, 'r') as file:
                self.config = json.load(file)
        else:
            print(f"Warning: Properties file '{properties_file}' not found. A new one will be created.")
            self.config = {}

    def get(self, key, default=None):
        return self.config.get(key, default)

    def display(self):
        for section, values in self.config.items():
            print(f"[{section}]")
            for key, value in values.items():
                print(f"{key} = {value}")
        print()
