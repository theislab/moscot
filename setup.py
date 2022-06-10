from pathlib import Path

from setuptools import setup, find_packages

try:
    from msc import __email__, __author__, __version__
except ImportError:
    __author__ = ""
    __version__ = "0.0.0"
    __email__ = ""


if __name__ == "__main__":
    setup(
        name="moscot",
        use_scm_version=True,
        setup_requires=["setuptools_scm"],
        author=__author__,
        author_email=__email__,
        maintainer=__author__,
        maintainer_email=__email__,
        version=__version__,
        description=Path("README.rst").read_text("utf-8").split("\n")[2],
        long_description=Path("README.rst").read_text("utf-8"),
        url="https://github.com/theislab/moscot",
        download_url="https://github.com/theislab/moscot",
        license="BSD",
        install_requires=Path("requirements.txt").read_text("utf-8").split("\n"),
        extras_require={
            "dev": ["pre-commit>=2.14.0", "tox>=3.24.0"],
        },
        zip_safe=False,
        packages=find_packages(),
        python_requires=">=3.7",
        platforms=["Linux", "MacOs", "Windows"],
        keywords=sorted(
            [
                "single-cell",
                "bio-informatics",
                "optimal transport",
            ]
        ),
        classifiers=[
            "Development Status :: 1 - Planning",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Natural Language :: English",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Typing :: Typed",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Scientific / Engineering :: Mathematics",
        ],
    )
