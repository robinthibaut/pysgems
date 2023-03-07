from setuptools import find_packages, setup

my_pckg = find_packages(include=["pysgems"],
                        exclude=["pysgems.deprecated", "pysgems.develop"])

# with open("README.rst") as f:
#     LONG_DESCRIPTION = f.read()

setup(
    name="pysgems",
    version="1.2.2",
    packages=my_pckg,
    include_package_data=True,
    # long_description=LONG_DESCRIPTION,
    long_description="Use SGeMS (Stanford Geostatistical Modeling Software) within Python.",
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
