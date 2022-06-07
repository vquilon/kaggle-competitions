import json


class SuperConfig(dict):
    @classmethod
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        self.__dict__ = self
        for k, v in self.items():
            setattr(self, k, v)
        return self

    @property
    def raw_json(self) -> str:
        return json.dumps(self)

    @staticmethod
    def get_dict_from(string: str) -> dict:
        return json.loads(string)

    def save_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.raw_json)

        return path

    def __str__(self) -> str:
        attributes_string = ", ".join([f"{k}={v}" for k, v in self.items()])
        return f"{self.__class__.__name__}({attributes_string})"

    __repr__ = __str__


class Config(dict):
    @classmethod
    def __new__(cls, *args, **kwargs):
        self = super(Config, cls).__new__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

        return self