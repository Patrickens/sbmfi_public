import sys
from sys import version_info
from setuptools import setup

if version_info[:2] < (3, 6):
    sys.stderr.write(f"unsupported python version: {version_info[:2]} should be 3.8 or higher")

if __name__ == "__main__":
    # https://gist.github.com/althonos/6914b896789d3f2078d1e6237642c35c NOTE: good example of cfg file
    setup(version="0.0.2")