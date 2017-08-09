from pip.req import parse_requirements
from setuptools import setup
from setuptools import find_packages

import sys
sys.path.append('.')
from deepcalcium import __version__
from deepcalcium.utils.config import get_config

# Create configuration.
get_config()


def ignore_req(reqstr):
    bad_reqs = ['c2s', 'cmt', 'tensorflow-gpu']
    return bool(sum([reqstr.startswith(b) for b in bad_reqs]))


# Some packages need to be installed directly from github.
# tensorflow-gpu cannot be installed with the explicit version.
full_reqs = parse_requirements('requirements.txt', session='hack')
full_reqs = [str(ir.req) for ir in full_reqs
             if not ignore_req(str(ir.req))]
full_reqs.append('tensorflow-gpu')

# Minimum requirements necessary to use deep-calcium as a library.
lib_reqs = ['numpy', 'scipy', 'cython', 'h5py', 'tqdm',
            'keras', 'scikit-image', 'sk-video', 'tensorflow',
            'neurofinder', 'pandas', 'requests']

setup(name='deepcalcium',
      version=__version__,
      description='Deep learning for calcium imaging data.',
      author='Alex Klibisz',
      author_email='aklibisz@gmail.com',
      url='https://github.com/alexklibisz/deep-calcium',
      install_requires=lib_reqs,
      extras_require={
          'develop': full_reqs
      },
      packages=find_packages())
