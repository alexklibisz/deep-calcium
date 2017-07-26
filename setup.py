from pip.req import parse_requirements
from setuptools import setup
from setuptools import find_packages

install_reqs = parse_requirements('requirements.txt', session='hack')

setup(name='deepcalcium',
      version='0.0.1',
      description='Deep learning for calcium imaging data.',
      author='Alex Klibisz',
      author_email='aklibisz@gmail.com',
      url='https://github.com/alexklibisz/deep-calcium',

      # Had trouble installing the explicit version tensorflow-gpu==1.2.1
      # from the requirements file, motivating the following fix.
      install_requires=['tensorflow-gpu'] +
          [str(ir.req) for ir in install_reqs
           if 'tensorflow-gpu' not in str(ir.req)],

      extras_require={},
      packages=find_packages())
