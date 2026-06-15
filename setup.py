"""Setup configuration for lite package."""

import re
from pathlib import Path

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip()]

init = Path("litekit") / "__init__.py"
version = re.search(
    r'__version__\s*=\s*["\']([^"\']+)["\']',
    init.read_text(encoding="utf-8"),
).group(1)

setup(
    name="litekit",
    version=version,
    author="Chaman Singh Verma",
    description="An unofficial, opinionated toolkit for LiteLLM (BerriAI) with vision and evaluation support.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cli-litetext=app.cli.liteclient_cli:main_cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
