import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "e19-co544-cricket-analytics-and-prediction"
AUTHOR_USER_NAME = "Bimbara28"
SRC_REPO = "CricVision"
AUTHOR_EMAIL = "e19324@eng.pdn.ac.lk"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Cricket Analytics and Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/cepdnaclk/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/cepdnaclk/{REPO_NAME}/issues",

    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
)
          