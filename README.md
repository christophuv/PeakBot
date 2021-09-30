# PeakBot

PeakBot is a python framework for peak-picking in LC-HRMS profile mode data.
It uses local-maxima in the LC-HRMS dataset each of which is then exported as a standarized two-dimensional area (rt x mz), which is used as the input for a machine-learning CNN model that reports whether the local-maxima is a chromatographic peak with left/right isomeric compounds or a signal of the background. Moreover, for chromatographic peaks it suggests a bounding-box and a peak-center.

![Workflow of PeakBot](https://github.com/christophuv/PeakBot/raw/main/workflow.png)

For training PeakBot offers some convenient functionality to extract reference features from the training chromatograms to help the user. This is achieved by first searching for chromatographic peaks using a smoothing and gradient-descend algorithm. The peaks' borders and centers are also estimated in this step. These peaks are then matched with a user-defined reference list (the ground-truth; isolated single chromatographic peaks) with the aim of using the same chromatographic peak but from different samples. In this step the properties of the reference features are also updated to best fit the chromatographic peaks from the reference chromatograms. Moreover, this pre-detection also allows to easily extend the reference list with isotopologs of the same compound.
The matched references are then used to generate a large number of training instances by iteratively combining them. Each such training instance can consist of a chromatographic peak or background signal and several other distraction peaks to generalize (augmentation of the training dataset). Moreover, also different background types are supported by PeakBot so that it differentiates between true chromatographic peaks and irrelevant background information (e.g., walls, which are signals present throughout the entire or large parts of the chromatograms). 

As a large number of training instances is required to train the CNN model and to achieve a high performance of the model, a GPU (CUDA) based approach is implemented that decreases the time required for their generation. 
The CNN model is implemented in the TensorFlow package (https://www.tensorflow.org/). It consists of several convolutional and pooling-layers and outputs a peak-type, -center, and -bounding-box. Examples of the detection can be exported to images for user-verification. Furthermore, the detected chromatographic peaks can also be exported as featureML files and tab-separated-values files. The featureML files are espeically helpful if the user wants to illustrate the detected features in for example TOPPView from the OpenMS toolbox (https://pubmed.ncbi.nlm.nih.gov/19425593/). 


## Run PeakBox
Several options are available to use PeakBot. The recommended one is a virtual environment with Anaconda. 

### GPU support
PeakBot uses the graphics processing unit of the PC for computational intensive tasks such as the generation of the large training dataset or the training of the CNN model. Thus, it requires a CUDA-enabled graphics card as well as the CUDA tookit and the cuDNN libraries to be installed. For further information about these packages please consult the official documentation of Nvidia at https://developer.nvidia.com/cuda-downloads, https://developer.nvidia.com/cudnn and https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html. 

#### Windows 10
On Windows 10 and newer the installation requires some additional steps. First, download the CUDA 11.2 toolbox and the cudnn 11.2 library from Nvidia and install them. Then start Nsight Monitor and enter the settings. There either disable the WDMM TDR or increase the delay to at least 30 seconds (depending on the used GPU).
Detailed instructions about the installation and configuration steps can be found at https://spltech.co.uk/how-to-install-tensorflow-2-5-with-cuda-11-2-and-cudnn-8-1-for-windows-10/ and https://docs.nvidia.com/gameworks/content/developertools/desktop/timeout_detection_recovery.htm. 

#### Linux 


### Install in a new conda virtual environment

- Make sure that you have Anaconda installed or install it from https://www.anaconda.com/.
- On Windows: start an Anaconda shell from the start menu
- Create a new conda virtual environment and activate it with the following commands:

```
conda create -n python3.8 python=3.8
conda activate python3.8
```
- Optional on Linux: Install GPU support with the command

```
conda install tensorflow-gpu
```

- Install the PeakBot framework with the command:

```
pip install git+https://github.com/christophuv/PeakBot
```


## Examples
Examples to train a new PeakBot-CNN model and to subsequently use if for the detection of new chromatographic peaks in other LC-HRMS chromatograms are available at [https://github.com/christophuv/PeakBot_Example](https://github.com/christophuv/PeakBot_Example)

