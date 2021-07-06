# PeakBot

PeakBot is a machine-learning CNN model for chromatographic peak picking in 2d profile mode LC-HRMS data.


## Install in a new conda virtual environment
Make sure you have Anaconda installed or install it. 

Create a new conda virtual environment and activate it
```
conda create -n 3.9 python=3.6
conda activate python3.9
```

Install the necessary packages
```
python -m pip install tensorflow
python -m pip install scipy
python -m pip install tqdm
python -m pip install numba
```