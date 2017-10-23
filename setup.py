#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import re

from setuptools import setup, find_packages


PKG_NAME = 'py-taxis'
README_PATH = 'README.md'


classifiers = """\
    Development Status :: 2 - Pre-Alpha
    Operating System :: OS Independent
    Intended Audience :: Science/Research
    License :: OSI Approved :: GNU General Public License v3 (GPLv3)
    Topic :: Scientific/Engineering :: Bio-Informatics
    Programming Language :: Python
    Programming Language :: Python :: 2
    Programming Language :: Python :: 2.7
"""


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop('encoding', 'utf-8')
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_version():
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        _read('{}/_version.py'.format(PKG_NAME)),
        re.MULTILINE).group(1)
    return version


def get_long_description():
    descr = _read(README_PATH)
    try:
        import pypandoc
        descr = pypandoc.convert_text(descr, to='rst', format='md')
    except (IOError, ImportError):
        pass
    return descr


install_requires = [
    'numpy>=1.9', 
    'pandas>=0.17',
    'trackpy>=0.3',
    'cv2>=3.1.0f',
    'sklearn'
    'hmmlearn'
]

python_requires = '~=2.7',


setup(
    name=PKG_NAME,
    author='Tatyana Perlova',
    author_email='perlova2@illinois.edu',
    version=get_version(),
    license='GPL-3.0',
    description='Python package for tracking and analysis of bacterial trajectories',
    long_description=get_long_description(),
    url='https://github.com/tatyana-perlova/py-taxis',
    keywords=['tracking', 'bacterial trajectories', 'taxis'],
    packages=find_packages(),
    zip_safe=False,
    classifiers=[s.strip() for s in classifiers.split('\n') if s],
    install_requires=install_requires,
    python_requires=python_requires,

    
)