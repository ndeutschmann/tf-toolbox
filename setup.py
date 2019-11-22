from setuptools import setup, find_packages

setup(
name='ml_toolbox',
version='0.4',
author='Nicolas Deutschmann',
author_email='nicolas.deutschmann@gmail.com',
packages=find_packages(),
url='https://github.com/ndeutschmann/tf-toolbox',
license='LICENSE',
description='Machine learning toolbox',
long_description=open('README.md').read(),
long_description_content_type='text/markdown',
   classifiers=[
      "Programming Language :: Python :: 3",
      "Operating System :: OS Independent",
   ],
install_requires=[
   "tensorflow",
    "tensorboard",
    "tqdm"
],
)