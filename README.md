# PeakBot

PeakBot is a python framework for peak-picking in LC-HRMS profile mode data.
It uses local-maxima in the LC-HRMS dataset each of which is then exported as a standarized two-dimensional area (rt x mz), which is used as the input for a machine-learning CNN model that reports whether the local-maxima is a chromatographic peak with left/right isomeric compounds or a signal of the background. Moreover, for chromatographic peaks it suggests a bounding-box and a peak-center.

![Workflow of PeakBot](https://github.com/christophuv/PeakBot/raw/main/workflow.png)

For training PeakBot offers some convenient functionality to extract reference features from the training chromatograms to help the user. This is achieved by first searching for chromatographic peaks using a smoothing and gradient-descend algorithm. The peaks' borders and centers are also estimated in this step. These peaks are then matched with a user-defined reference list (the ground-truth; isolated single chromatographic peaks) with the aim of using the same chromatographic peak but from different samples. In this step the properties of the reference features are also updated to best fit the chromatographic peaks from the reference chromatograms. Moreover, this pre-detection also allows to easily extend the reference list with isotopologs of the same compound.
The matched references are then used to generate a large number of training instances by iteratively combining them. Each such training instance can consist of a chromatographic peak or background signal and several other distraction peaks to generalize (augmentation of the training dataset). Moreover, also different background types are supported by PeakBot so that it differentiates between true chromatographic peaks and irrelevant background information (e.g., walls, which are signals present throughout the entire or large parts of the chromatograms). 

As a large number of training instances is required to train the CNN model and to achieve a high performance of the model, a GPU (CUDA) based approach is implemented that decreases the time required for their generation. 
The CNN model is implemented in the TensorFlow package (https://www.tensorflow.org/). It consists of several convolutional and pooling-layers and outputs a peak-type, -center, and -bounding-box. Examples of the detection can be exported to images for user-verification. Furthermore, the detected chromatographic peaks can also be exported as featureML files and tab-separated-values files. The featureML files are espeically helpful if the user wants to illustrate the detected features in for example TOPPView from the OpenMS toolbox (https://pubmed.ncbi.nlm.nih.gov/19425593/). 


## Install PeakBox
PeakBot is a pyhton package and can thus be run on different operating system. However, the recommended method for installation is to run it in a virtual environment with Anaconda as all CUDA dependencies can automatically be installed there. 

### GPU support
PeakBot uses the graphics processing unit of the PC for computational intensive tasks such as the generation of the large training dataset or the training of the CNN model. Thus, it requires a CUDA-enabled graphics card from Nvidia as well as the CUDA tookit and the cuDNN libraries to be installed. For further information about these packages please consult the official documentation of Nvidia at https://developer.nvidia.com/cuda-downloads, https://developer.nvidia.com/cudnn and https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html. 
If the Anaconda environment together with a virtual environment is used, these steps can be omitted as they can easily be installed there.

Note: Different GPUs have a different number of streaming-processors. Thus, the blockdim and griddim need to be chosen accordingly. Please adapt these values to your GPU. To find good values for your particular GPU, the script quickFindCUDAParameters.py from the PeakBot examples repository can be used. It iterates over possible combinations of the parameters and tries to detect peaks with these. When all have been tested, a list with the best is presented and these parameters should then be used.
Note: If an exportBatchSize of 2048 requires some 4GB of GPU-memory. If you have less, try reducing this value to 1024 of 512. 

### Windows 10
1. Update your Nvidia GPU driver to the latest available. Driveras can be downloaded from https://www.nvidia.com/Download/index.aspx. 
2. Increase the WDMM TDR (timeout for GPU refresh). This can be done via the Registry Editor. For this, plese press the Windows key and enter regedit. Do not start it, but right-click on the entry "Registry Editor" and start it as an administrator by clicking "Run as Administrator". Enter the credentials of your administrator account and start the Registry Editor. Navigate to the directory "HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers" and create the key "TdrLevel" as a REG_DWORD with the value 3. Add another key "TdrDelay" with the value 45. This will increase the timeout for the display refresh to 45 seconds. Then restart your system. More information about these to keys can be found at https://www.pugetsystems.com/labs/hpc/Working-around-TDR-in-Windows-for-a-better-GPU-computing-experience-777/ and https://msdn.microsoft.com/en-us/library/windows/hardware/ff569918(v=vs.85).aspx.
3. Download and install Anaconda from https://www.anaconda.com. 
4. Start the "Anaconda prompt" from the star menu. 
5. Create a new conda virtual environment and activate it with the following commands:

```
    conda create -n python3.8 python=3.8
    conda activate python3.8
```

6. Install some dependencies (Git, Cuda toolkit and cudnn libraries)

```
    conda install git
    conda install cudnn==8.2.1
    conda install -c anaconda urllib3
```

7. Install the PeakBot framework with the command:

```
    pip install git+https://github.com/christophuv/PeakBot
```

8. Fix for newer CUDA dlls. Copy the file "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\cusolver64_11.dll" to "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\cusolver64_10.dll". (Fix by https://stackoverflow.com/a/65608751).


9. Optional: Download a sample file and run it. For this, navigate to a new directory in the command prompt and enter the commands: 

```
    curl https://raw.githubusercontent.com/christophuv/PeakBot_Example/main/quickExample.py > quickExample.py
    python quickExample.py
```


### Linux 
1. Update your Nvidia GPU driver to the latest available. Drivers can be downloaded from https://www.nvidia.com/Download/index.aspx. 
2. Install the Anaconda environment. Instructions are available at https://docs.anaconda.com/anaconda/install/linux/
3. Start a bash-shell
4. Install git via

```
    sudo apt update
    sudo apt install git
```
5. Create a new conda virtual environment and activate it with the following commands:

```
    conda create -n python3.8 python=3.8
    conda activate python3.8
```

6. Install some dependencies (Git, Cuda toolkit and cudnn libraries)

```
    conda install git
    conda install cudnn==8.2.1
```

7. Install the PeakBot framework with the command:

```
    pip install git+https://github.com/christophuv/PeakBot
```

8. Optional: Download a sample file and run it. For this, navigate to a new directory in the command prompt and enter the commands: 

```
    wget https://raw.githubusercontent.com/christophuv/PeakBot_Example/main/quickExample.py
    python quickExample.py
```


## Examples
Examples to train a new PeakBot-CNN model and to subsequently use if for the detection of new chromatographic peaks in other LC-HRMS chromatograms are available at https://github.com/christophuv/PeakBot_Example. 

