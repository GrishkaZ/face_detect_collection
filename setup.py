from setuptools import setup, find_packages

from face_detect_collection import __version__

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='face_detect_collection',
    version=__version__,

    url='https://github.com/GrishkaZ/face_detect_collection',
    author='GZ',
    author_email='grigoriozaly@yandex.ru',
    packages=find_packages(),
    install_requires=requirements,
    package_data={
        "face_detect_collection": [
            # 'models/face_landmarker.task',
            'models/face_detection_models/*',
            'models/face_detection_models/face_detector_caffe/*',
            ],
    },
    # include_package_data=True,
)