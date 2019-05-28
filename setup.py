from setuptools import find_packages, setup

setup(name='recogners',
      version='1.0.0',
      description='Python-Inspired Machine Learners',
      author='Recogna Laboratory',
      author_email='recogna@fc.unesp.br',
      url='https://github.com/recogna-lab/recogners',
      license='MIT',
      install_requires=['coverage>=4.5.2',
                        'pylint>=1.7.4',
                        'pytest>=3.2.3'
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
          'Programming Language :: PyPy :: 3.5',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
