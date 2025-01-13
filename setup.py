from setuptools import find_packages, setup
from distutils.util import convert_path

def read_requirements(filename: str):
    with open(filename) as requirements_file:
        requirements = []
        for line in requirements_file:
            line = line.strip()
    return requirements

main_ns = {}
ver_path = convert_path('src/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)


setup(
    name="cmu-11967-hw1",
    author='Yiming Zhang, Clement Fung, Xinyue Liu',
    author_email='yimingz3@cs.cmu.edu',
    version=main_ns['__version__'],
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
