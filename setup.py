#!/usr/bin/env python

__author__ = "Christopher Hahne"
__email__ = "inbox@christopherhahne.de"
__license__ = """
    Copyright (c) 2022 Christopher Hahne <inbox@christopherhahne.de>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from setuptools import setup, find_packages
from torchimize import __version__
from docutils import core
from pathlib import Path


path = Path.cwd()
# parse description section text
with open(str(path / 'README.rst'), 'r') as f:
    data = f.read()
    readme_nodes = list(core.publish_doctree(data))
    for node in readme_nodes:
        if node.astext().startswith('Description'):
                long_description = node.astext().rsplit('\n\n')[1]

# parse package requirements from text file
with open(str(path / 'requirements.txt'), 'r') as f:
    req_list = f.read().split('\n')

setup(
      name='torchimize',
      version=__version__,
      description='Optimization Algorithms using Pytorch',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      url='http://github.com/hahnec/torchimize',
      author='Christopher Hahne',
      author_email='inbox@christopherhahne.de',
      license='GNU GPL V3.0',
      keywords='pytorch torch optimization mathematical linear programming gauss newton levenberg marquardt',
      packages=find_packages(),
      install_requires=req_list,
      include_package_data=True,
      zip_safe=False,
      )