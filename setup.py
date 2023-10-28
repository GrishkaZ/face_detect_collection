from setuptools import setup

from face_detect_collection import __version__

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='face_detect_collection',
    version=__version__,

    url='https://github.com/GrishkaZ/face_detect_collection',
    author='GZ',
    author_email='grigoriozaly@yandex.ru',
    packages=['face_detect_collection'],
    install_requires=requirements,
)