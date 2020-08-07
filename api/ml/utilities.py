from pathlib import Path
import yaml

def get_config():
    with open('config.yml') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)
    return config

def file_handler(object_type, model_type=None):
    config = get_config()
    parent_file = Path(__file__).parent.parent
    if model_type:
        extension = config['file_paths'][object_type][model_type]
    else:
        extension = config['file_paths'][object_type]
    return parent_file / extension