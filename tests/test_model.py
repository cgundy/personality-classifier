import pytest
from personality_classifier.ml.model_pipeline import train, predict, get_accuracy
from personality_classifier.ml.model_pipeline import get_config

config = get_config()

# Todo: create test data


@pytest.mark.parametrize("model_type", config["valid_models"])
def test_predict(model_type):
    assert predict(["Hello I'm Carly"], model_type)[0].lower() in config["types"]


@pytest.mark.parametrize("model_type", config["valid_models"])
def test_accuracy(model_type):
    assert 0 <= get_accuracy(model_type) <= 1
