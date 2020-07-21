from setuptools import find_packages, setup


TESTS_REQUIRE = [
    "pytest",
    # Needed by tests.test_env_wrapper, but not DRLHP proper
    "stable_baselines~=2.10",
]


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
     name='drlhp',
     version='0.1',
     scripts=[],
     author="Matthew Rahtz, Updated by Cody Wild",
     author_email="codywild@berkeley.edu",
     description="More general implementation of Deep RL from Human Preferences",
     install_requires=[
         "scipy>=1.3.2",
         "matplotlib>=3.0.0",
         "gym[atari]>=0.14.0",
         "easy-tf-log==1.1",
         "tensorflow~=1.15.2",
     ],
     extras_require={
        "dev": TESTS_REQUIRE,
     },
     tests_require=TESTS_REQUIRE,
     dependency_links=['http://github.com/mrahtz/gym-moving-dot/tarball/master#egg=gym-moving-dot'],
     long_description=long_description,
     packages=find_packages(exclude=["tests"]),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
