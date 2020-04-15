from setuptools import find_packages, setup
from setuptools.extension import Extension
import numpy 
import os


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires

if __name__ == '__main__':
    setup(
        name='billTemplate',
        packages=find_packages(exclude=('notebook', 'test','snapshot','sample')),
        license='Apache License 2.0',
#         setup_requires=['pytest-runner', 'cython', 'numpy'],
#         tests_require=['pytest'],
        install_requires=get_requirements(),
        ext_modules=[
            Extension('billTemplate.common.template.utils.compute_overlap', ['billTemplate/common/template/utils/compute_overlap.pyx'],
                      include_dirs=[numpy.get_include()])],
    setup_requires = ["cython>=0.28", "numpy>=1.14.0"])
