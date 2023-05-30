from setuptools import find_packages, setup

requirements = []

# Read requirements from files
with open("requirements.txt", "r") as f:
    requirements.extend(f.read().splitlines())

setup(
    name="PeachDB",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
    dependency_links=["https://download.pytorch.org/whl/cu113"],
)
