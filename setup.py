from setuptools import setup, find_packages

setup(
    name="gym-jumper",
    version="0.0.1",
    packages=find_packages("."),
    install_requires=["gym", "stable-baselines3"]
)
