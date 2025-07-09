from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements-test.txt") as f:
    test_requirements = f.read().splitlines()

setup(
    name="tennis_predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    tests_require=test_requirements,
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Powered Tennis Match Prediction System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tennis-predictor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "tennis-predictor=app:main",
        ],
    },
)
