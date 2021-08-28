from setuptools import setup, find_packages

DISTNAME = "octopy"
VERSION = "1.0.0"
DESCRIPTION = """Octopy is a companion Python library for octosport.io. The library offers tools for football (soccer) analytics."""
LONG_DESCRIPTION = """Octopy is a companion Python library for octosport.io. The library offers tools for football (soccer) analytics."""
AUTHOR = "octosport.io"
AUTHOR_EMAIL = "contact@octosport.io"
URL = "https://github.com/octosport/octopy"
LICENSE = "Apache License, Version 2.0"


REQUIREMENTS = ["pandas>=1.1.3", "scipy>=1.5.2", "scikit-learn>=0.23.2", "jax>=0.2.17"]

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        packages=find_packages(),
        package_data={"docs": ["*"]},
        include_package_data=True,
        zip_safe=False,
        install_requires=REQUIREMENTS,
        classifiers=["Programming Language :: Python :: 3.4"],
    )
