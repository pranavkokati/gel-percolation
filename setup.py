from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="gelrigidity",
    version="0.2.0",
    description=(
        "Dual-rigidity-percolation load-path-continuity model for enzymatically "
        "degrading hydrogel wound scaffolds undergoing simultaneous ECM deposition"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pranav Kokati",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["gelrigidity", "gelrigidity.*"]),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "networkx>=3.1",
        "pandas>=2.0.0",
        "joblib>=1.3.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
