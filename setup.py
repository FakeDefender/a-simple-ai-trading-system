from setuptools import find_packages, setup


setup(
    name="trading_ai",
    version="0.1.0",
    description="A practical baseline AI-assisted trading research system",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0.0",
        "yfinance>=0.2.40",
        "requests>=2.31.0",
    ],
    python_requires=">=3.10",
)
