from setuptools import find_packages, setup

my_pckg = find_packages(include=["pysgems"],
                        exclude=["pysgems.deprecated", "pysgems.develop"])

setup(
    name="pysgems",
    version="1.1.8",
    packages=my_pckg,
    include_package_data=True,
    url="https://github.com/robinthibaut/pysgems",
    license="MIT",
    author="Robin Thibaut",
    author_email="robin.thibaut@UGent.be",
    description="Use SGeMS (Stanford Geostatistical Modeling Software) within Python.",
    install_requires=["numpy", "pandas", "scipy", "matplotlib", "loguru"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
