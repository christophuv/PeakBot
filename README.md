# PeakBot

PeakBot is a python framework for peak-picking in LC-HRMS profile mode data.
It uses local-maxima in the LC-HRMS dataset each of which is then exported as a standarized two-dimensional area (rt x mz), which is used as the input for a machine-learning CNN model that reports whether the local-maxima is a chromatographic peak with left/right isomeric compounds or a signal of the background. Moreover, for chromatographic peaks it suggests a bounding-box and a peak-center.

For training PeakBot offers some convenient functionality to test for chromatographic peaks using a smoothing and gradient-descend algorithm to estimate the peaks' borders and centers. These peaks are then matched with a user-defined reference list (the ground-truth; isolated single chromatographic peaks) with the aim of using the same chromatographic peak but from different samples. Moreover, this pre-detection also allows to easily extend the reference list with isotopologs of the same compound.
The matched references are then used to generate a large number of training instances by compining them. Each such training instance can consist of a chromatographic peak or background signal and several other distraction peaks to generalize (augmentation of the training dataset).
As a large number of such training instances is required to train the CNN model, a GPU (CUDA) based approach is implemented that decreases the time required for their generation.
The CNN model is implemented in the TensorFlow package (https://www.tensorflow.org/). It consists of several convolutional and pooling-layers and outputs a peak-type, -center, and -bounding-box. Examples of the detection can be exported to images for user-verification.

The output of PeakBot is a tsv-file of detected profile-mode chromatographic peaks as well as a featureML file that can be used to visualize the detection process in TOPPView (https://www.openms.de/).


## Run PeakBox
Several options are available to use PeakBot. The preferred one is a virtual environment with Anaconda

### Install in a new conda virtual environment
Make sure you have Anaconda installed or install it (https://www.anaconda.com/).
Create a new conda virtual environment and activate it
```
conda create -n python3.8 python=3.8
conda activate python3.8
```

#### Windows 10
On Windows 10 and newer the installation requires some additional steps. Download the CUDA 11.2 toolbox and the cudnn 11.2 library and install them. Then start Nsight Monitor and enter the settings. There either disable the WDMM TDR or increase the delay to at least 30 seconds (depending on the used CPU).
Detailed instructions about the installation and configuration steps can be found at https://spltech.co.uk/how-to-install-tensorflow-2-5-with-cuda-11-2-and-cudnn-8-1-for-windows-10/ and https://docs.nvidia.com/gameworks/content/developertools/desktop/timeout_detection_recovery.htm. 
