"""
TURBO_gift Setup
================

Install with: pip install -e .

This makes the package properly importable and eliminates IDE warnings.
"""

from setuptools import setup, find_packages

setup(
    name="turbo_gift",
    version="1.0.0",
    author="M.L. McKnight",
    author_email="maesonsfarms@gmail.com",
    description="A free GPU/CPU optimization toolkit - A Gift to Humanity",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ml-innovations/turbo_gift",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "web": ["flask>=2.0.0"],
        "torch": ["torch>=1.9.0"],
        "tensorflow": ["tensorflow>=2.6.0"],
        "dev": ["pytest>=6.0.0"],
        "all": [
            "flask>=2.0.0",
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
            "pytest>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
    ],
    keywords="gpu cpu optimization memory performance machine-learning",
    license="MIT",
)
