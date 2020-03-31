#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Pillow >= 7.0.0',
    'barbar >= 0.2.1',
    'jpeg4py >= 0.1.4',
    'matplotlib >= 3.0.0',
    'numpy >= 1.18.1',
    'opencv_python >= 4.2.0.32',
    'pandas >= 0.25.0',
    'scikit_learn >= 0.22.1',
    'scikit_plot >= 0.3.7',
    'torch >= 1.4.0',
    'torchvision >= 0.5.0']

#     'Cython',
#     'pycocotools >= 2.0.0',

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Zac Winzurk",
    author_email='zwinzurk@asu.edu',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python package for computer vision on camera trap images.",
    entry_points={
        'console_scripts': [
            'torchtraps=torchtraps.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='torchtraps',
    name='torchtraps',
    packages=find_packages(include=['torchtraps', 'torchtraps.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/winzurk/torchtraps',
    version='0.1.4',
    zip_safe=False,
)
