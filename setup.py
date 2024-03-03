from setuptools import setup, find_packages

setup(
    name="giraffe",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "matplotlib",
        "graphviz",
        "paretoset",
        "tinygrad @ git+https://github.com/tinygrad/tinygrad",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
