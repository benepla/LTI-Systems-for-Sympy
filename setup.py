#!/usr/bin/env python

from distutils.core import setup

setup(name='Lti Systems for python',
      version='0.1.0',
      description='LTI control system package for symbolic python',
      author='Benedikt Placke',
      author_email='benedikt.placke@outlook.com',
      url='https://github.com/m3zz0m1x/LTI-Systems-for-Sympy',
      packages=['lti_systems'],
      requires=['sympy', 'scipy']
      )
