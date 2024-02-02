from setuptools import setup, find_packages

setup(
    name = "hela",
    version = "1.0.0",
    description="package untuk memudahkan perhitungan",
    author="arfy slowy",
    author_email="slowy.arfy@contoh.mail.com",
    package_data={"Hela": ["util/*"]},
    packages=find_packages()
)