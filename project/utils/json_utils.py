import json
import os

class JsonUtils:

    @staticmethod
    def read_json(file_path: str):
        if not os.path.isfile(file_path):
            return {}
        if not file_path.endswith('.json'):
            return {}
        with open(file_path, 'rb') as fp:
            return json.load(fp)