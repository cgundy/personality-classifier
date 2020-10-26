from pathlib import Path
import yaml
import pickle
import numpy as np
import logging
from typing import List, Set, Dict, Tuple, Optional, Union, Any


def get_config() -> Dict:
    """Load the config file"""
    with open("config.yml") as c:
        config = yaml.load(c, Loader=yaml.FullLoader)
    return config


def set_logger() -> logging.Logger:
    """Create logger and set log level to info"""
    logger = logging.getLogger("my-logger")
    logger.setLevel(logging.INFO)
    return logger


def file_handler(object_type: str, model_type: Optional[str] = None) -> str:
    """Find the correct file path for a given input"""
    config = get_config()
    parent_file = Path(__file__).parent.parent / "ml"
    if model_type:
        extension = config["file_paths"][object_type][model_type]
        file_path = parent_file / extension
    else:
        file_path = config["file_paths"][object_type]
    return file_path


def save(object, object_type: str, model_type: Optional[str] = None) -> None:
    """Save a given object"""
    file_path = file_handler(object_type, model_type)
    with open(file_path, "wb") as infile:
        pickle.dump(object, infile)


def load(object_type: str, model_type: Optional[str] = None) -> Any:
    """Load a given object"""
    file_path = file_handler(object_type, model_type)
    with open(file_path, "rb") as outfile:
        return pickle.load(outfile)
