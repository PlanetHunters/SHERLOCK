import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()
version = "0.24.1-RC1"

setuptools.setup(
    name="sherlockpipe", # Replace with your own username
    version=version,
    author="F.J. Pozuelos & M. Dévora",
    author_email="fjpozuelos@uliege.be",
    description="Search for Hints of Exoplanets fRom Lightcurves Of spaCe based seeKers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/franpoz/SHERLOCK",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.8',
    install_requires=['astroplan==0.7',
                        "astroquery==0.4.1",
                        "allesfitter==1.2.2",
                        'argparse==1.4.0',
                        "celerite==0.4.0", # Allesfitter dependency
                        "cython==0.29.21",
                        "extension-helpers==0.1",
                        "requests==2.25.1",
                        "matplotlib==3.3.4",
                        "pyyaml==5.4.1",
                        "dynesty==1.0.1",
                        "emcee==3.0.2",
                        "corner==2.1.0",
                        "ellc==1.8.5",
                        "seaborn==0.11.1",
                        "bokeh==2.2.3",
                        "sklearn==0.0",
                        "reproject==0.4",
                        "reportlab==3.5.59",
                        "rebound==3.17.3",
                        "spock==1.3.1",
                        "lcbuilder==0.4.0",
                        "mock==4.0.3",
                        'tqdm==4.56.0',
                        'setuptools>=41.0.0',
                        'torch==1.7.1',
                        'beautifulsoup4==4.9.3',
                        'numba>=0.53.0rc1',
                        'configparser==5.0.1',
                        'pyparsing==2.4.7', # Matplotlib dependency
                        'statsmodels==0.12.2', # Allesfitter dependency
                        'triceratops==1.0.8',
                        'uncertainties==3.1.5'
    ]
)