import json

class FileUtils:
    """文件工具包
    """
    @staticmethod
    def read_json(file_path: str):
        try:
            with open(file_path, 'r') as fp:
                return json.load(fp)
        except:
            return {}

    @staticmethod
    def write_json(data, file_path: str):
        with open(file_path, 'w') as fp:
            json.dump(data, fp)