from setuptools import setup, find_packages

setup(
    name="personality-classifier",
    packages=find_packages(),
    version="0.1.0",
    entry_points={"console_scripts": ["train = personality-classifier:train"]},
)