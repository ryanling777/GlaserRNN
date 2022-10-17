from setuptools import setup, find_packages

setup(
    name='modular_rnn',
    version='0.0.1',
    packages = find_packages(),
    install_requires=[
        'numpy',
        'torch'
    ],
)

