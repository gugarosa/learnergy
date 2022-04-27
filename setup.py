from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="learnergy",
    version="1.1.3",
    description="Energy-based Machine Learners",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mateus Roder and Gustavo de Rosa",
    author_email="mateus.roder@unesp.br, gustavo.rosa@unesp.br",
    url="https://github.com/gugarosa/learnergy",
    license="Apache 2.0",
    install_requires=[
        "coverage>=5.5",
        "matplotlib>=3.3.4",
        "Pillow>=8.1.2",
        "pre-commit>=2.17.0",
        "pylint>=2.7.2",
        "pytest>=6.2.2",
        "requests>2.23.0",
        "scikit-image>=0.17.2",
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "tqdm>=4.49.0",
    ],
    extras_require={
        "tests": [
            "coverage",
            "pytest",
            "pytest-pep8",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)
