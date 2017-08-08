from pip.req import parse_requirements
from setuptools import setup
from setuptools import find_packages

# Some packages need to be installed directly from github.
# tensorflow-gpu cannot be installed with the explicit version.


def ignore_req(reqstr):
    bad_reqs = ['c2s', 'cmt', 'tensorflow-gpu']
    return bool(sum([reqstr.startswith(b) for b in bad_reqs]))


full_reqs = parse_requirements('requirements.txt', session='hack')
full_reqs = [str(ir.req) for ir in full_reqs
             if not ignore_req(str(ir.req))]
full_reqs.append('tensorflow-gpu')

setup(name='deepcalcium',
      version='0.0.1',
      description='Deep learning for calcium imaging data.',
      author='Alex Klibisz',
      author_email='aklibisz@gmail.com',
      url='https://github.com/alexklibisz/deep-calcium',
      install_requires=['numpy', 'scipy', 'h5py',
                        'keras', 'scikit-image', 'tensorflow'],
      extras_require={
          'develop': full_reqs
      },
      packages=find_packages())
