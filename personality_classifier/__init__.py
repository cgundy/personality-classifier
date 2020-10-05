from .ml.model_pipeline import train, predict, get_accuracy
import click
import logging
from .ml.utilities import get_config, set_logger


config = get_config()


def _train_model(logger: logging.Logger, model_type: str) -> None:
    logger.info(f"Training {model_type}")
    train(model_type)
    accuracy = get_accuracy(model_type)
    logger.info(f"Model training for {model_type} complete with {accuracy} accuracy.")


def train_all() -> None:
    logger = set_logger()
    for model_type in config["valid_models"]:
        _train_model(logger, model_type)


@click.command()
@click.option("--model_type")
def train_model(model_type: str) -> None:
    logger = set_logger()
    _train_model(logger, model_type)


def get_prediction():
    pass
