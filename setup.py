from setuptools import setup, find_packages

setup(
    name="lutipy",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    author="Ranit Karmakar",
    author_email="ranit@hms.harvard.edu",
    description="A python package to automatically create figures for multi-channel microscopy images using complementary LUTs.",
    url="https://github.com/rkarmaka/lutipy",  # Replace with your repository URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
