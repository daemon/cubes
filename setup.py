import setuptools


setuptools.setup(
    name="cubes",
    version="0.0.1",
    author="Ralph Tang",
    author_email="r33tang@uwaterloo.ca",
    description="CUDA kernels with Python bindings for deep learning applications.",
    install_requires=["pynvrtc"],
    include_package_data=True,
    url="https://github.com/daemon/cubes",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)