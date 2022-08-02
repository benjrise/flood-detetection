import dataclasses
import yaml
from pathlib import Path

@dataclasses.dataclass
class Config:
    EXTRA : str
    NUM_CLASSES : int
    PRETRAINED_PATH : str

def dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name:f.type for f in dataclasses.fields(klass)}
        return klass(**{f:dataclass_from_dict(fieldtypes[f],d[f]) for f in d})
    except:
        return d # Not a dataclass field

def get_hrnet_config(config_path):
    conf = yaml.safe_load(Path(config_path).read_text())
    conf = dataclass_from_dict(Config, conf)
    return conf

def get_hrnet_weights():
    pass