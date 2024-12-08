import yaml

class Config:
    def __init__(self, conf_file):
        with open(conf_file, "r") as f:
            self._conf = yaml.safe_load(f)

    def __getattr__(self, name):
        return self._conf.get(name)