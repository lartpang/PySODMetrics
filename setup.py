from setuptools import setup, find_packages

setup(
    name="pysodmetrics",
    packages=find_packages(),
    version="1.2.1",
    license="MIT",
    description="A simple and efficient implementation of SOD metrics.",
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
