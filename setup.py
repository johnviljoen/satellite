"""
generate a python env with version 3.9, and pip install -e . on this and u
should be good to go.
"""

from setuptools import setup, find_packages

setup(name='satellite',
    version='0.0.1',
    url='https://gitlab.pnnl.gov/dadaist/dpc_safety_filter/',
    author='John Viljoen',
    author_email='johnviljoen2@gmail.com',
    install_requires=[
        'neuromancer',
        'casadi'
    ],
    packages=find_packages(
        include=[
            'satellite',
            'satellite.*'
        ]
    ),
)

