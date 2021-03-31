# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("./version.txt", encoding="utf-8", mode="r") as f:
    version = f.readline().strip()

with open("readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pysodmetrics",
    packages=find_packages(),
    version=version,
    license="MIT",
    description="A simple and efficient implementation of SOD metrics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="lartpang",
    author_email="lartpang@gmail.com",
    url="https://github.com/lartpang/PySODMetrics",
    keywords=[
        "salient object detection",
        "saliency detection",
        "metric",
        "deep learning",
    ],
    install_requires=["scipy>=1.5,<2", "numpy>=1.18,<2"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
