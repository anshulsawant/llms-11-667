from setuptools import find_packages, setup
from distutils.util import convert_path

def read_requirements(filename: str):
    with open(filename) as requirements_file:
        requirements = []
        for line in requirements_file:
            line = line.strip()
    return requirements

setup(
    name="cmu-11967-hw12",
    author='Anshul Sawant',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=read_requirements("requirements.txt"),
    python_requires="~=3.11",
    entry_points = {
        'pytest11': [
            'pytest_utils = pytest_utils.pytest_plugin',
        ]
    },
)
