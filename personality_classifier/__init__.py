from .ml.model_pipeline import train, predict, get_accuracy
import logging
from .ml.utilities import get_config


config = get_config()


def train_all():
    logger = logging.getLogger("my-logger")
    logger.setLevel(logging.INFO)
    for model_type in config["valid_models"]:
        logger.info(f"Training {model_type}")
        train(model_type)
        accuracy = get_accuracy(model_type)
        logger.info(
            f"Model training for {model_type} complete with {accuracy} accuracy."
        )


def get_prediction():
    pass
