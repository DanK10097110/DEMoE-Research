#!/usr/bin/env python
"""Setup configuration for DEMoE package."""

from setuptools import setup, find_packages

setup(
    name="demoe",
    version="1.0.0",
    description="DEMoE: Distributed Experts Mixture of Experts",
    packages=find_packages(include=["demoe*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
        ],
    },
)
