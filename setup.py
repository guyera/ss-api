#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines()]

setup_requirements = [
    "setuptools",
]

test_requirements = ["pytest"]

setup(
    author="PAR Government and Kitware, Inc.",
    author_email="adam_kaufman@partech.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="""Framework for deploying and evaluating novelty detection
                    algorithms for DARPA Sail-On.""",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    name="sail-on-ss",
    packages=find_packages(exclude=["tests"]),
    setup_requires=setup_requirements,
    install_requires=requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://gitlab.kitware.com/darpa-sail-on/sail-on",
    version="0.0.2",
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "sail_on_server_ss=sail_on.api.server:command_line",
        ]
        # TODO: set up new scripts for calling test_ids_request
    },
)
