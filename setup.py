import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    # Application name:
    name='python-handwritten-digit-gen',

    # Version number (initial):
    version='0.1',
    
    #desctiption
    description='Handwritten digit generator written in python',
    long_description=readme(),
	
    # Application author details:
    author="Aki Kutvonen",
    author_email="aki.kutvonen@gmail.com",

    # Packages
    packages=["digitgenerator"],

    # Include additional files into the package
    include_package_data=True,

    scripts=['bin/generate_png'],
    test_suite='nose.collector',
    tests_require=['nose'],
	
    # Details
    url="https://github.com/akutvone/python-handwritten-digit-gen",

    # licence
    license="LICENSE.txt",
    
    classifiers=[
	    'Development Status :: 3 - Alpha',
	    'Intended Audience :: Developers',
	    'Operating System :: OS Independent',
	    'Programming Language :: Python',
	    'Topic :: Scientific/Engineering :: Image Recognition'],
          

    # Dependent packages (distributions)
    install_requires=[
        'numpy>=1.10',
	'matplotlib>=1.40'
    ],
)
