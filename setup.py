from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='learnergy',
      version='1.1.0',
      description='Energy-based Machine Learners',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Mateus Roder and Gustavo de Rosa',
      author_email='mateus.roder@unesp.br, gustavo.rosa@unesp.br',
      url='https://github.com/gugarosa/learnergy',
      license='GPL-3.0',
      install_requires=['coverage>=5.3',
                        'matplotlib>=3.3.2',
                        'Pillow>=7.2.0',
                        'pylint>=2.6.0',
                        'pytest>=6.1.0',
                        'torch>=1.6.0',
                        'torchvision>=0.7.0',
                        'tqdm>=4.50.0'
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
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
