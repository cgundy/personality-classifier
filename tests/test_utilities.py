from typing import Dict
from personality_classifier.ml.utilities import get_config


def test_get_config():
    config = get_config()
    assert isinstance(config["types"], list)
    assert isinstance(config["model_parameters"]["tfidf"], Dict)
