from setuptools import find_packages, setup

setup(
    name="sentiment-analysis-rnn",
    version="2.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.12.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "pyyaml>=6.0",
        "streamlit>=1.22.0",
        "pandas>=1.5.0",
    ],
)
