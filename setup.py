import pathlib

from setuptools import find_packages, setup


def _get_long_description():
    path = pathlib.Path(__file__).parent / 'README.md'
    with open(path, encoding='utf-8') as f:
        long_description = f.read()
    return long_description


def _get_requirements(path):
    with open(path) as f:
        data = f.readlines()
    return data


setup(
    name='anime-face-detector',
    version='0.0.9',
    author='hysts',
    url='https://github.com/hysts/anime-face-detector',
    python_requires='>=3.7',
    install_requires=_get_requirements('requirements.txt'),
    packages=find_packages(exclude=('tests', )),
    include_package_data=True,
    description='Anime Face Detector using mmdet and mmpose',
    long_description=_get_long_description(),
    long_description_content_type='text/markdown',
)
