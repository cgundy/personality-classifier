from setuptools import setup, find_packages

setup(
    name="personality_classifier",
    packages=find_packages(),
    version="0.0.1",
    entry_points={
        "console_scripts": [
            "train_all = personality_classifier:train_all",
            "train_model = personality_classifier:train_model",
        ]
    },
)