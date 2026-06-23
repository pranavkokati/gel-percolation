from setuptools import setup, find_packages

setup(
    name="gel-percolation",
    version="0.1.0",
    description=(
        "Percolation Inversion Dynamics in Enzymatically Degrading Wound Hydrogels: "
        "A Topological Early Warning Framework for Predicting Fibroblast Invasion Windows"
    ),
    author="Pranav Kokati",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "networkx>=3.1",
        "pandas>=2.0.0",
        "joblib>=1.3.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "ripser>=0.6.4",
        "persim>=0.3.3",
        "gudhi>=3.8.0",
        "seaborn>=0.12.0",
        "mesa>=2.1.0",
    ],
    entry_points={
        "console_scripts": [
            "gel-percolation=run_simulation:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
