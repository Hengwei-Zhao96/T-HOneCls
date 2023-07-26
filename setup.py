from setuptools import setup
from setuptools import find_packages


VERSION = '0.0.0'

setup(
    name='HOC',  # package name
    version=VERSION,  # package version
    description='Hyperspectral One Class Classification',  # package description
    packages=find_packages(),
    zip_safe=False,
)