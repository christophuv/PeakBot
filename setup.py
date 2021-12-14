#from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='peakbot',
    version='0.4.6',
    author='Christoph Bueschl',
    author_email='christoph.bueschl [the little email symbol] univie.ac.at',
    packages=find_packages(),#['peakbot', 'peakbot.train'],
    url='https://github.com/christophuv/PeakBot',
    license='LICENSE',
    description='A machine-learning CNN model for peak picking in profile mode LC-HRMS data',
    long_description=open('README.md').read(),
    install_requires=[
        "tensorflow == 2.4.1",
        "numba >= 0.53.1",
        "pandas >= 1.2.3",
        "matplotlib >= 3.4.2",
        "plotnine >= 0.8.0",

        "pymzml", 

        "tqdm >= 4.61.2",
        "natsort >= 7.1.1",    
        "py-cpuinfo >= 8.0.0",
        "psutil"
    ],
)
