"""Setup configuration for multilingual-entity-resolver."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multilingual-entity-resolver",
    version="1.0.0",
    author="Shahin Hasanov",
    author_email="apucalip@gmail.com",
    description="Multilingual company name resolver for customs trade documents using fuzzy matching and semantic similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShahinHasanov90/multilingual-entity-resolver",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
