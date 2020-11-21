from setuptools import find_packages, setup


setup(
    name="tts",
    version="0.0.1",
    author="isadrtdinov",
    package_dir={"": "tts"},
    packages=find_packages("tts"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

