
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=2.2.0', 'gym==0.17.1', 'opencv-python==4.2.0.32', 'atari-py==0.2.6', 'pyyaml>=5.3.1', 'hyperopt>=0.2.4']

setup(
    name='pong_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
