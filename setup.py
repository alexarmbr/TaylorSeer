from setuptools import setup, find_packages

setup(
    name="taylor-seer",
    version="0.1.0",
    description="A decorator for approximating neural network forward passes using Taylor series",
    author="Alex Armbruster",
    packages=find_packages(),
    install_requires=["torch"],
    extras_require={
        "dev": ["pytest"],
    },
) 