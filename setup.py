from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='recogners',
      version='1.0.1',
      description='Python-Inspired Machine Learners',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Recogna Laboratory',
      author_email='recogna@fc.unesp.br',
      url='https://github.com/recogna-lab/recogners',
      license='MIT',
      install_requires=['coverage>=4.5.2',
                        'matplotlib>=3.1.0',
                        'pandas>=0.24.2',
                        'Pillow>=6.0.0',
                        'pylint>=1.7.4',
                        'pytest>=3.2.3',
                        'torch>=1.1.0',
                        'torchvision>=0.3.0'
                       ],
      extras_require={
          'tests': ['coverage',
                    'pytest',
                    'pytest-pep8',
                   ],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
