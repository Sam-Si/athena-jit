from setuptools import setup, find_packages

setup(
    name="athena-jit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "llvmlite==0.43.0",
        "numpy>=2.0.0",
        "psutil",
    ],
)
