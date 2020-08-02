import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="irrigation30",  # Replace with your own username
    version="0.0.1",
    author="Will Hawkins",
    author_email="whawkins@berkeley.edu",
    description="Generate irrigation predictions at 30m resolution using Google Earth Engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wmhawkins-iv/irrigation30",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
