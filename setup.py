#!/usr/bin/env python
import codecs
import os.path
from distutils.core import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(name='BayesCCal',
      version=get_version("BayesCCal/__init__.py"),
      description='Bayesian calibration of classifiers',
      long_description='Bayesian calibration of classifiers',
      author='Marco Puts',
      author_email='mputs@acm.org',
      packages=['BayesCCal'],
      install_requires=['numpy', 'scipy'],
     )
