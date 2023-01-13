#!/usr/bin/env python

from distutils.core import setup

setup(name='rlll',
      version='1.0',
      description='Reinforcement Learning for Lunar Landing',
      author='Pablo Campillo Sanchez',
      author_email='dev@pablocampillo.pro',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=['rlll'],
      package_dir={'rlll': 'src/rlll'},
     )