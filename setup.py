from setuptools import setup, find_packages

setup(
    name="Hela",
    version="1.0.0",
    description="package untuk memudahkan untuk berhitung",
    author="arfy slowy",
    package_data={"Hela": ["util/*"]},
    python_requires=">=3.10",
    packages=find_packages()
)