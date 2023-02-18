#!/usr/bin/env python

import os
import setuptools

def get_long_description():
    filename = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(filename) as f:
        return f.read()

setuptools.setup(name='gestop',
      version='1.0.1',
      description="Navigate Desktop with Gestures",
      long_description=get_long_description(),
      long_description_content_type="text/markdown",
      author='Sriram Krishna, Nishant Sinha',
      url='https://github.com/ofnote/gestop',
      license='Apache 2.0',
      platforms=['POSIX'],
      packages=setuptools.find_packages(),
      package_data={'': ['data/*', 'models/*']},
      include_package_data=True,
      #entry_points={},
      #scripts=[],
      classifiers=[
          'Environment :: Console',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
          ],
      install_requires=['mediapipe', 
                        'matplotlib',
                        'pandas',
                        'protobuf',
                        'pynput',
                        'pytorch-lightning',
                        'scikit-learn',
                        'torch', 
                        'torchvision',
                        'wandb'
                        ]
      )
