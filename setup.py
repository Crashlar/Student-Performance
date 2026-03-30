from setuptools import setup, find_packages
from pathlib import Path
from typing import  List

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


HYPEN_E_DOT = "-e ."
def get_requirements(filename : str) -> List[str]:
    """
    this function is used to get the requirements from a requirements file
    :param filename:
    :return:
    """
    requirements = []
    with open(filename) as file:
        requirements = file.readlines()
        requirements=[req.replace("\n" , "") for req in requirements ]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="studentperformance",
    version="0.1.0",

    author="Mukesh Kumar",
    author_email="mukeshkumar.in25@gmail.com",

    description="this is a student performance based project that tells the performance on using of student data ",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/Crashlar/Student-Performance.git",

    license="MIT",

    packages=find_packages(),
    # package_dir={"": "src"},

    # install_requires=[
    #     "numpy",
    #     "pandas",
    #     "seaborn",
    #     "scikit-learn"
    # ],
    # else this is better
    install_requires = get_requirements("requirements.txt"),

    entry_points={
        "console_scripts": [
            "studentperformance=app:main"
        ]
    },
    classifiers=[],

    keywords="student performance,  performance",

    python_requires=">=3.11",
    include_package_data=True,
)