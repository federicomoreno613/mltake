from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="takehome",
    version="1",
    packages=find_packages(),
    author="Federico Moreno",
    author_email="federicomoreno613@gmail.com",
    description="Takehome project ",
    url="https://github.com/federicomoreno613/mltake",
    scripts=['scripts/takehome-run'],
    install_requires=requirements
)
