from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='learnergy',
      version='1.0.6',
      description='Energy-based Machine Learners',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Mateus Roder and Gustavo de Rosa',
      author_email='mateus.roder@unesp.br, gustavo.rosa@unesp.br',
      url='https://github.com/gugarosa/learnergy',
      license='GPL-3.0',
      install_requires=['coverage>=5.1',
                        'matplotlib>=3.2.1',
                        'Pillow>=7.1.2',
                        'pylint>=2.5.2',
                        'pytest>=5.4.2',
                        'torch>=1.5.0',
                        'torchvision>=0.6.0',
                        'tqdm>=4.46.0'
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
