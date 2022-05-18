from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="sdat2",
    author_email="author@example.com",
    description="Scripts to run emulation of ADCIRC/SWAN in the vacinity of New Orleans",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
