from setuptools import setup, find_packages

setup(
    name="gym_sokoban",
    version="0.0.1",
    install_requires=["scipy", "pillow", "imageio"],
    packages=["gym_sokoban"]
)