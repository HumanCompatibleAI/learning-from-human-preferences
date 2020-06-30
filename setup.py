from setuptools import setup


with open("README.md", "r") as fh:
    long_description = "More general implementation of Deep RL from Human Preferences" # fh.read() Temporarily modified due to ascii issue 

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
     name='drlhp',
     version='0.1',
     scripts=[],
     author="Matthew Rahtz, Updated by Cody Wild",
     author_email="codywild@berkeley.edu",
     description="More general implementation of Deep RL from Human Preferences",
     install_requires=requirements,
     dependency_links=['http://github.com/mrahtz/gym-moving-dot/tarball/master#egg=gym-moving-dot'],
     long_description=long_description,
     packages=['drlhp'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
