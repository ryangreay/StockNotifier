from setuptools import setup, find_packages

setup(
    name="stock_notifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.28",
        "python-dotenv>=1.0.0",
        "pushbullet.py>=0.12.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-jose>=3.3.0",
        "joblib>=1.3.0",
        "pytest>=7.4.0"
    ]
) 