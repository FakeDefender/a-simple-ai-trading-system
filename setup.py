from setuptools import setup, find_packages

setup(
    name="trading_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "crewai",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "torch",
        "transformers",
        "python-dotenv",
        "pyyaml"
    ],
    python_requires=">=3.8",
) 