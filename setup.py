from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="learnergy",
    version="1.2.0",
    description="Energy-based Machine Learners",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mateus Roder and Gustavo de Rosa",
    author_email="mateus.roder@unesp.br, gustavo.rosa@unesp.br",
    url="https://github.com/gugarosa/learnergy",
    license="Apache 2.0",
    install_requires=[
        "matplotlib>=3.3.4",
        "Pillow>=8.1.2",
        "requests>=2.23.0",
        "scikit-image>=0.17.2",
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "tqdm>=4.49.0",
    ],
    extras_require={
        "dev": [
            "pre-commit>=2.17.0",
        ],
        "tests": [
            "coverage",
            "pylint",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)
