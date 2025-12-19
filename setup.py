"""
Setup script for yield_curves package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="yield_curves",
    version="0.1.0",
    author="Brett Cooper",
    description="Bond yield curve analysis and currency swap evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yield_curves",  # Update with your repo
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "openpyxl>=3.1",
        "pandera>=0.17",
        "pydantic>=2.0",
        "nelson-siegel-svensson>=0.4.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "black>=23.0",
            "flake8>=6.0",
            "jupyter>=1.0",
            "jupyterlab>=4.0",
        ],
    },
)
