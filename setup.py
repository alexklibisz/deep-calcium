from pip.req import parse_requirements
from setuptools import setup
from setuptools import find_packages

# Some packages need to be installed directly from github.
# tensorflow-gpu cannot be installed with the explicit version.


def ignore_req(reqstr):
    bad_reqs = ['c2s', 'cmt', 'tensorflow-gpu']
    return bool(sum([reqstr.startswith(b) for b in bad_reqs]))


install_reqs = parse_requirements('requirements.txt', session='hack')
install_reqs = [str(ir.req) for ir in install_reqs
                if not ignore_req(str(ir.req))]
install_reqs.append('tensorflow-gpu')

setup(name='deepcalcium',
      version='0.0.1',
      description='Deep learning for calcium imaging data.',
      author='Alex Klibisz',
      author_email='aklibisz@gmail.com',
      url='https://github.com/alexklibisz/deep-calcium',
      install_requires=install_reqs,
      extras_require={},
      packages=find_packages())
