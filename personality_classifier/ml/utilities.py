from pathlib import Path
import yaml
import pickle
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union, Any


def get_config() -> Dict:
    with open("config.yml") as c:
        config = yaml.load(c, Loader=yaml.FullLoader)
    return config


def file_handler(object_type: str, model_type: Optional[str] = None) -> str:
    config = get_config()
    parent_file = Path(__file__).parent.parent / "ml"
    if model_type:
        extension = config["file_paths"][object_type][model_type]
    else:
        extension = config["file_paths"][object_type]
    return parent_file / extension


def save(object, object_type: str, model_type: Optional[str] = None) -> None:
    file_path = file_handler(object_type, model_type)
    with open(file_path, "wb") as infile:
        pickle.dump(object, infile)


def load(object_type: str, model_type: Optional[str] = None) -> Any:
    file_path = file_handler(object_type, model_type)
    with open(file_path, "rb") as outfile:
        return pickle.load(outfile)
