import logging

from .core import tic, toc, tocAddStat, timeit, writeTSVFile

import os
import datetime
import pickle
import uuid

import tensorflow as tf
import numpy as np
import numba
import scipy
import pandas as pd
import plotnine as p9

import tqdm




#####################################
### Configuration class
##
class Config(object):
    """Base configuration class"""

    NAME    = "PeakBot"
    VERSION = "0.9"

    RTSLICES       =  32   ## should be of 2^n
    MZSLICES       = 128   ## should be of 2^m
    NUMCLASSES     =   6   ## [isFullPeak, hasCoelutingPeakLeftAndRight, hasCoelutingPeakLeft, hasCoelutingPeakRight, isWall, isBackground]
    FIRSTNAREPEAKS =   4   ## specifies which of the first n classes represent a chromatographic peak (i.e. if classes 0,1,2,3 represent a peak, the value for this parameter must be 4)

    BATCHSIZE     =  32
    STEPSPEREPOCH =  64
    EPOCHS        = 512

    DROPOUT        = 0.2
    UNETLAYERSIZES = [32,64,128,256]

    LEARNINGRATESTART              = 0.005
    LEARNINGRATEDECREASEAFTERSTEPS = 5
    LEARNINGRATEMULTIPLIER         = 0.9
    LEARNINGRATEMINVALUE           = 3e-7

    INSTANCEPREFIX = "___PBsample_"

    @staticmethod
    def getAsStringFancy():
        return "\n  | ..".join([
            "  | .. %s"%(Config.NAME),
            " Version " + Config.VERSION,
            " Size of LC-HRMS area: %d x %d (rts x mzs)"%(Config.RTSLICES, Config.MZSLICES),
            " Number of peak-classes: %d"%(Config.NUMCLASSES),
            " Batchsize: %d, Epochs %d, StepsPerEpoc: %d"%(Config.BATCHSIZE, Config.EPOCHS, Config.STEPSPEREPOCH),
            " DropOutRate: %g"%(Config.DROPOUT),
            " UNetLayerSizes: %s"%(Config.UNETLAYERSIZES),
            " LearningRate: Start: %g, DecreaseAfter: %d steps, Multiplier: %g, min. rate: %g"%(Config.LEARNINGRATESTART, Config.LEARNINGRATEDECREASEAFTERSTEPS, Config.LEARNINGRATEMULTIPLIER, Config.LEARNINGRATEMINVALUE),
            " Prefix for instances: '%s'"%Config.INSTANCEPREFIX,
            " "
        ])

    @staticmethod
    def getAsString():
        return ";".join([
            "%s"%(Config.NAME),
            "Version " + Config.VERSION,
            "Size of LC-HRMS area: %d x %d (rts x mzs)"%(Config.RTSLICES, Config.MZSLICES),
            "Number of peak-classes: %d"%(Config.NUMCLASSES),
            "Batchsize: %d, Epochs %d, StepsPerEpoc: %d"%(Config.BATCHSIZE, Config.EPOCHS, Config.STEPSPEREPOCH),
            "DropOutRate: %g"%(Config.DROPOUT),
            "UNetLayerSizes: %s"%(Config.UNETLAYERSIZES),
            "LearningRate: Start: %g, DecreaseAfter: %d steps, Multiplier: %g, min. rate: %g"%(Config.LEARNINGRATESTART, Config.LEARNINGRATEDECREASEAFTERSTEPS, Config.LEARNINGRATEMULTIPLIER, Config.LEARNINGRATEMINVALUE),
            "InstancePrefix: '%s'"%(Config.INSTANCEPREFIX),
            ""
        ])



print("Initializing PeakBot")
try:
    import platform
    print("  | .. OS:", platform.platform())
except Exception:
    print("  | .. fetching OS information failed")

print("  | .. TensorFlow version: %s"%(tf.__version__))

try:
    import cpuinfo
    s = cpuinfo.get_cpu_info()["brand_raw"]
    print("  | .. CPU: %s"%(s))
except Exception:
    print("  | .. fetching CPU info failed")

try:
    from psutil import virtual_memory
    mem = virtual_memory()
    print("  | .. Main memory: %.1f GB"%(mem.total/1000/1000/1000))
except Exception:
    print("  | .. fetching main memory info failed")

try:
    from numba import cuda as ca
    print("  | .. GPU-device: ", str(ca.get_current_device().name), sep="")

    gpus = tf.config.experimental.list_physical_devices()
    for gpu in gpus:
        print("  | .. TensorFlow device: Name '%s', type '%s'"%(gpu.name, gpu.device_type))
except Exception:
    print("  | .. fetching GPU info failed")











#####################################
### Data generator methods
### Read files from a directory and prepare them
### for PeakBot training and prediction
##
def dataGenerator(folder, instancePrefix = None, verbose=False):

    if instancePrefix is None:
        instancePrefix = Config.INSTANCEPREFIX

    ite = 0
    while os.path.isfile(os.path.join(folder, "%s%d.pickle"%(instancePrefix, ite))):
        
        l = pickle.load(open(os.path.join(folder, "%s%d.pickle"%(instancePrefix, ite)), "rb"))
        assert all(np.amax(l["LCHRMSArea"], (1,2)) == 1), "LCHRMSarea is not scaled to a maximum of 1 '%s'"%(str(np.amax(l["LCHRMSArea"], (1,2))))
        yield l
        ite += 1

def modelAdapterTrainGenerator(datGen, newBatchSize = None, verbose=False):
    ite = 0
    l = next(datGen)
    while l is not None:

        if verbose and ite == 0:
            logging.info("  | Generated data is")

            for k, v in l.items():
                if type(v).__module__ == "numpy":
                    logging.info("  | .. gt: %18s numpy: %30s %10s"%(k, v.shape, v.dtype))
                else:
                    logging.info("  | .. gt: %18s  type:  %40s"%(k, type(v)))
            logging.info("  |")

        if newBatchSize is not None:

            for k in l.keys():
                if   isinstance(l[k], np.ndarray) and len(l[k].shape)==2:
                    l[k] = l[k][0:newBatchSize,:]

                elif isinstance(l[k], np.ndarray) and len(l[k].shape)==3:
                    l[k] = l[k][0:newBatchSize,:,:]

                elif isinstance(l[k], np.ndarray) and len(l[k].shape)==4:
                    l[k] = l[k][0:newBatchSize,:,:,:]

                elif isinstance(l[k], list):
                    l[k] = l[k][0:newBatchSize]

        yield {"LCHRMSArea": l["LCHRMSArea"]}, \
              {"peakType" : l["peakType"],
               "center"   : l["center"  ],
               "box"      : l["box"     ],
               #"single"   : l["single"  ],
              }
        l = next(datGen)
        ite += 1

def modelAdapterTestGenerator(datGen, newBatchSize = None, verbose=False):
    return modelAdapterTrainGenerator(datGen, newBatchSize, verbose)














#####################################
### PeakBot additional methods
##
def iou(boxes1, boxes2):
    ## from https://github.com/paperclip/cat-camera/blob/master/object_detection_2/core/post_processing.py
    """Calculates the overlap between proposal and ground truth boxes.
    Some `boxes2` may have been padded. The returned `iou` tensor for these
    boxes will be -1.
    Args:
    boxes1: a tensor with a shape of [batch_size, N, 4]. N is the number of
    proposals before groundtruth assignment. The last dimension is the pixel
    coordinates in [ymin, xmin, ymax, xmax] form.
    boxes2: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
    tensor might have paddings with a negative value.
    Returns:
    iou: a tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
    """
    with tf.name_scope('BatchIOU'):
        x1_min, x1_max, y1_min, y1_max = tf.split(value=boxes1, num_or_size_splits=4, axis=1)
        x2_min, x2_max, y2_min, y2_max = tf.split(value=boxes2, num_or_size_splits=4, axis=1)

        # Calculates the intersection area
        intersection_xmin = tf.maximum(x1_min, x2_min)
        intersection_xmax = tf.minimum(x1_max, x2_max)
        intersection_ymin = tf.maximum(y1_min, y2_min)
        intersection_ymax = tf.minimum(y1_max, y2_max)
        intersection_area = tf.maximum(tf.subtract(intersection_xmax, intersection_xmin), 0) * tf.maximum(tf.subtract(intersection_ymax, intersection_ymin), 0)

        # Calculates the union area
        area1 = tf.multiply(tf.subtract(y1_max, y1_min), tf.subtract(x1_max, x1_min))
        area2 = tf.multiply(tf.subtract(y2_max, y2_min), tf.subtract(x2_max, x2_min))
        # Adds a small epsilon to avoid divide-by-zero.
        union_area = tf.add(tf.subtract(tf.add(area1, area2), intersection_area), tf.constant(1e-8))

        # Calculates IoU
        iou = tf.divide(intersection_area, union_area)

        return iou

## adapted from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def recall(y_true, y_pred):
    true_positives     = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall_keras

## adapted from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def precision(y_true, y_pred):
    true_positives      = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision_keras

## adapted from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def specificity(y_true, y_pred):
    tn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + tf.keras.backend.epsilon())

## adapted from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def negative_predictive_value(y_true, y_pred):
    tn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + tf.keras.backend.epsilon())

## adapted from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

def pF1(y_true, y_pred):
    return f1(tf.cast(tf.math.less(tf.math.argmax(y_true, axis=1), Config.FIRSTNAREPEAKS), tf.float32),
              tf.cast(tf.math.less(tf.math.argmax(y_pred, axis=1), Config.FIRSTNAREPEAKS), tf.float32))
def pTPR(y_true, y_pred):
    return recall(tf.cast(tf.math.less(tf.math.argmax(y_true, axis=1), Config.FIRSTNAREPEAKS), tf.float32),
                  tf.cast(tf.math.less(tf.math.argmax(y_pred, axis=1), Config.FIRSTNAREPEAKS), tf.float32))
def pFPR(y_true, y_pred):
    return 1-precision(tf.cast(tf.math.less(tf.math.argmax(y_true, axis=1), Config.FIRSTNAREPEAKS), tf.float32),
                       tf.cast(tf.math.less(tf.math.argmax(y_pred, axis=1), Config.FIRSTNAREPEAKS), tf.float32))




def convertGeneratorToPlain(gen, numIters=1):
    x = None
    y = None
    for i, t in enumerate(gen):
        if i < numIters:
            if x is None:
                x = t[0]
                y = t[1]
            else:
                for k in x.keys():
                    x[k] = np.concatenate((x[k], t[0][k]), axis=0)
                for k in y.keys():
                    y[k] = np.concatenate((y[k], t[1][k]), axis=0)
        else:
            break
    return x,y

## modified from https://stackoverflow.com/a/47738812
## https://github.com/LucaCappelletti94/keras_validation_sets
class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, logDir, validation_sets=None, verbose=0, batch_size=None, steps=None, everyNthEpoch=1):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.logDir = logDir
        self.validation_sets = []
        if validation_sets is not None:
            for validation_set in validation_sets:
                self.addValidationSet(validation_set)
        self.verbose = verbose
        self.batch_size = batch_size
        self.steps = steps
        self.everyNthEpoch = everyNthEpoch
        self.lastEpochNum = 0
        self.history = []
        self.printWidths = {}
        self.maxLenNames = 0

    def addValidationSet(self, validation_set):
        if len(validation_set) not in [3, 4]:
            raise ValueError()
        self.validation_sets.append(validation_set)


    @timeit
    def on_epoch_end(self, epoch, logs=None, ignoreEpoch = False):
        self.lastEpochNum = epoch
        hist = None
        if epoch%self.everyNthEpoch == 0 or ignoreEpoch:
            hist={}
            if self.verbose: print("Additional test datasets (epoch %d): "%(epoch+1))
            # evaluate on the additional validation sets
            for validation_set in self.validation_sets:
                tic("kl234hlkjsfkjh1hlkjhasfdkjlh")
                outStr = []
                if len(validation_set) == 2:
                    validation_data, validation_set_name = validation_set
                    validation_targets = None
                    sample_weights = None
                if len(validation_set) == 3:
                    validation_data, validation_targets, validation_set_name = validation_set
                    sample_weights = None
                elif len(validation_set) == 4:
                    validation_data, validation_targets, sample_weights, validation_set_name = validation_set
                else:
                    raise ValueError()

                results = self.model.evaluate(x=validation_data,
                                              y=validation_targets,
                                              verbose=False,
                                              sample_weight=sample_weights,
                                              batch_size=self.batch_size,
                                              steps=self.steps)

                self.maxLenNames = max(self.maxLenNames, len(validation_set_name))

                file_writer = tf.summary.create_file_writer(self.logDir + "/" + validation_set_name)
                for i, (metric, result) in enumerate(zip(self.model.metrics_names, results)):
                    valuename = "epoch_" + metric
                    with file_writer.as_default():
                        tf.summary.scalar(valuename, data=result, step=epoch)
                    if i > 0: outStr.append(", ")
                    valuename = metric
                    if i not in self.printWidths.keys():
                        self.printWidths[i] = 0
                    self.printWidths[i] = max(self.printWidths[i], len(valuename))
                    outStr.append("%s: %.4f"%("%%%ds"%self.printWidths[i]%valuename, result))
                    hist[validation_set_name + "_" + valuename] = result
                outStr.append("")
                outStr.insert(0, "   %%%ds: %3.0f"%(self.maxLenNames, toc("kl234hlkjsfkjh1hlkjhasfdkjlh"))%validation_set_name)
                if self.verbose: print("".join(outStr))
            if self.verbose: print("")
        self.history.append(hist)

    def on_train_end(self, logs=None):
        self.on_epoch_end(self.lastEpochNum, logs=logs, ignoreEpoch = True)

class PeakBot():
    def __init__(self, name, ):
        super(PeakBot, self).__init__()

        batchSize = Config.BATCHSIZE
        rts = Config.RTSLICES
        mzs = Config.MZSLICES
        numClasses = Config.NUMCLASSES
        version = Config.VERSION

        self.name       = name
        self.batchSize  = batchSize
        self.rts        = rts
        self.mzs        = mzs
        self.numClasses = numClasses
        self.model      = None
        self.version    = version


    @timeit
    def buildTFModel(self, verbose = False):
        dropOutRate = Config.DROPOUT
        uNetLayerSizes = Config.UNETLAYERSIZES

        if verbose:
            print("  | PeakBot v %s model"%(self.version))
            print("  | .. Desc: Detection of a single LC-HRMS peak in an area")
            print("  | ")

        ## Input: Only LC-HRMS area
        input_ = tf.keras.Input(shape=(self.rts, self.mzs, 1), name="LCHRMSArea")
        if verbose:
            print("  | .. Inputs")
            print("  | .. .. LCHRMSArea is", input_)
            print("  |")

        ## Encoder
        x = input_
        cLayers = [x]
        for i in range(len(uNetLayerSizes)):

            x = tf.keras.layers.ZeroPadding2D(padding=2)(x)
            x = tf.keras.layers.Conv2D(uNetLayerSizes[i], (5,5), use_bias=True)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

            #x = tf.keras.layers.Dropout(dropOutRate)(x)
            x = tf.keras.layers.ZeroPadding2D(padding=1)(x)
            x = tf.keras.layers.Conv2D(uNetLayerSizes[i], (3,3), use_bias=True)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPool2D((2,2))(x)
            cLayers.append(x)
        lastUpLayer = x

        ## Intermediate layer and feature properties (indices and borders)
        x = tf.keras.layers.BatchNormalization()(x)
        fx          = tf.keras.layers.Flatten()(x)
        peakType    = tf.keras.layers.Dense(self.numClasses, name="peakType", activation="sigmoid")(fx)
        center      = tf.keras.layers.Dense(              2, name="center"  , activation="relu"   )(fx)
        box         = tf.keras.layers.Dense(              4, name="box"     , activation="relu"   )(fx)

        #tf.math.argmax(peakType, axis=1)
        #b_broadcast = tf.zeros(tf.shape(center), dtype=center.dtype)
        #center = tf.where(tf.greater(center, 3), b_broadcast, center)
        #b_broadcast = tf.zeros(tf.shape(box), dtype=box.dtype)
        #box = tf.where(tf.greater(box, 3), b_broadcast, box, name="box")


        ### Decoder
        #for i in range(len(uNetLayerSizes)-1, -1, -1):
        #    x = tf.keras.layers.UpSampling2D((2,2))(x)
        #    x = tf.concat([x, cLayers[i]], axis=3)
        #    x = tf.keras.layers.ZeroPadding2D(padding=2)(x)
        #    x = tf.keras.layers.Conv2D(uNetLayerSizes[i-1] if i > 0 else 1, (5,5), use_bias=True)(x)
        #    x = tf.keras.layers.BatchNormalization()(x)
        #    x = tf.keras.layers.Activation("relu")(x)
        #
        #    x = tf.keras.layers.ZeroPadding2D(padding=1)(x)
        #    x = tf.keras.layers.Conv2D(uNetLayerSizes[i-1] if i > 0 else 1, (3,3), use_bias=True)(x)
        #    x = tf.keras.layers.BatchNormalization()(x)
        #    if i > 0:
        #        x = tf.keras.layers.Add()([x, cLayers[i]])
        #    x = tf.keras.layers.Activation("relu")(x)

        #x = tf.keras.layers.ZeroPadding2D(padding=1)(x)
        #x = tf.keras.layers.Conv2D(1, (3,3))(x)
        #x = tf.keras.layers.BatchNormalization()(x)

        #x = tf.keras.layers.Activation("sigmoid", name="single")(x)
        #single = x

        if verbose:
            print("  | .. Intermediate layer")
            print("  | .. .. lastUpLayer is", lastUpLayer)
            print("  | .. .. fx          is", fx)
            print("  | .. ")
            print("  | .. Outputs")
            print("  | .. .. peak        is", peakType)
            print("  | .. .. center      is", center)
            print("  | .. .. box         is", box)
            #print("  | .. .. single      is", single)
            print("  | .. ")
            print("  | ")

        self.model = tf.keras.models.Model(input_, [peakType, center, box]) #, single])

    @timeit
    def compileModel(self, learningRate = None):
        if learningRate is None:
            learningRate = Config.LEARNINGRATESTART
        cce = tf.keras.losses.CategoricalCrossentropy()
        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate),
            loss         = {
                        "peakType" : "CategoricalCrossentropy",
                        "center"   : tf.keras.losses.Huber(),
                        "box"      : tf.keras.losses.Huber(),
                        #"single"   : "MSE",
                      },
            metrics      = {
                        "peakType" : ["categorical_accuracy", pF1, pTPR, pFPR],
                        "center"   : [],
                        "box"      : [iou],
                      },
            loss_weights = {
                        "peakType" : 1,
                        "center"   : 1,
                        "box"      : 1,
                        #"single"   : 1,
                      }
        )

    @timeit
    def train(self, datTrain, datVal, logDir = None, callbacks = None, verbose = 1):
        epochs = Config.EPOCHS
        steps_per_epoch = Config.STEPSPEREPOCH

        if verbose:
            print("  | Fitting model on training data")
            print("  | .. Logdir is '%s'"%logDir)
            print("  | .. Number of epochs %d"%(epochs))
            print("  |")

        if logDir is None:
            logDir = os.path.join("logs", "fit", "PeakBot_v" + self.version + "_" + uuid.uuid4().hex)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logDir, histogram_freq = 1, write_graph = True)
        file_writer = tf.summary.create_file_writer(os.path.join(logDir, "scalars"))
        file_writer.set_as_default()

        _callBacks = []
        _callBacks.append(tensorboard_callback)
        if callbacks is not None:
            if type(callbacks) is list:
                _callBacks.extend(callbacks)
            else:
                _callBacks.append(callbacks)

        # Fit model
        history = self.model.fit(
            datTrain,
            validation_data = datVal,

            batch_size      = self.batchSize,
            epochs          = epochs,
            steps_per_epoch = steps_per_epoch,

            callbacks = _callBacks,

            verbose = verbose
        )

        return history

    def loadFromFile(self, modelFile):
        self.model = tf.keras.models.load_model(modelFile, custom_objects = {"iou": iou})

    def saveModelToFile(self, modelFile):
        self.model.save(modelFile)







@timeit
def trainPeakBotModel(trainInstancesPath, logBaseDir, modelName = None, valInstancesPath = None, addValidationInstances = None, everyNthEpoch = 15, verbose = False):
    tic("pbTrainNewModel")

    ## name new model
    if modelName is None:
        modelName = "%s%s__%s"%(Config.NAME, Config.VERSION, uuid.uuid4().hex[0:6])

    ## log information to folder
    logDir = os.path.join(logBaseDir, modelName)
    logger = tf.keras.callbacks.CSVLogger(os.path.join(logDir, "clog.tsv"), separator="\t")

    if verbose:
        print("Training new PeakBot model")
        print("  | Model name is '%s'"%(modelName))
        print("  | .. config is")
        print(Config.getAsStringFancy().replace(";", "\n"))
        print("  |")

    ## define learning rate schedule
    def lrSchedule(epoch, lr):
        if (epoch + 1) % Config.LEARNINGRATEDECREASEAFTERSTEPS == 0:
            lr *= Config.LEARNINGRATEMULTIPLIER
        tf.summary.scalar('learningRate', data=lr, step=epoch)

        return max(lr, Config.LEARNINGRATEMINVALUE)
    lrScheduler = tf.keras.callbacks.LearningRateScheduler(lrSchedule, verbose=False)

    ## create generators for training data (and validation data if available)
    datGenTrain = modelAdapterTrainGenerator(dataGenerator(trainInstancesPath, verbose = verbose), newBatchSize = Config.BATCHSIZE, verbose = verbose)
    datGenVal   = None
    if valInstancesPath is not None:
        datGenVal = modelAdapterTestGenerator(dataGenerator(valInstancesPath  , verbose = verbose), newBatchSize = Config.BATCHSIZE, verbose = verbose)
        datGenVal = tf.data.Dataset.from_tensors(next(datGenVal))

    ## add additional validation datasets to monitor model performance during the trainging process
    valDS = AdditionalValidationSets(logDir,
                                     batch_size=Config.BATCHSIZE,
                                     everyNthEpoch = everyNthEpoch,
                                     verbose = verbose)
    if addValidationInstances is not None:
        if verbose:
            print("  | Additional validation datasets")
        for valInstance in addValidationInstances:
            x,y = None, None
            if "folder" in valInstance.keys():
                print("Adding", valInstance["folder"])
                datGen  = modelAdapterTestGenerator(dataGenerator(valInstance["folder"]),
                                                    newBatchSize = Config.BATCHSIZE)
                numBatches = valInstance["numBatches"] if "numBatches" in valInstance.keys() else 1
                x,y = convertGeneratorToPlain(datGen, numBatches)
            if "x" in valInstance.keys():
                x = valInstance["x"]
                y = valInstance["y"]
            if x is not None and y is not None and "name" in valInstance.keys():
                valDS.addValidationSet((x,y, valInstance["name"]))
                if verbose:
                    print("  | .. %s: %d instances"%(valInstance["name"], x["LCHRMSArea"].shape[0]))

            else:
                raise RuntimeError("Unknonw additional validation dataset")
        if verbose:
            print("  |")

    ## instanciate a new model and set its parameters
    pb = PeakBot(modelName)
    pb.buildTFModel(verbose = verbose)
    pb.compileModel(learningRate = Config.LEARNINGRATESTART)

    ## train the model
    history = pb.train(
        datTrain = datGenTrain,
        datVal   = datGenVal,

        logDir = logDir,
        callbacks = [logger, lrScheduler, valDS],

        verbose = verbose * 2
    )

    ## save metrices of the training process in a user-convenient format (pandas table)
    metricesAddValDS = pd.DataFrame(columns=["model", "set", "metric", "value"]);
    if addValidationInstances is not None:
        hist = valDS.history[-1]
        for valInstance in addValidationInstances:

            se = valInstance["name"]
            for metric in ["loss", "peakType_loss", "center_loss", "box_loss", "peakType_categorical_accuracy", "peakType_pF1", "peakType_pTPR", "peakType_pFPR", "box_iou"]:
                val = hist[se + "_" + metric]
                newRow = pd.Series({"model": modelName, "set": se, "metric": metric, "value": val})
                metricesAddValDS = metricesAddValDS.append(newRow, ignore_index=True)

    if verbose:
        print("  |")
        print("  | .. model built and trained successfully (took %.1f seconds)"%toc("pbTrainNewModel"))

    return pb, metricesAddValDS
















@timeit
def runPeakBot(pathFrom, modelPath, verbose = True):
    tic("detecting with peakbot")

    peaks = []
    backgrounds = []
    walls = []
    errors = []

    peaksDone = 0
    debugPrinted = 0
    if verbose:
        print("Detecting peaks with PeakBot")
        print("  | .. loading PeakBot model '%s'"%(modelPath))
        print("  | .. detecting peaks in the areas in the folder '%s'"%(pathFrom))

    model = tf.keras.models.load_model(modelPath, custom_objects = {"iou": iou,"recall": recall,
                                                                    "precision": precision, "specificity": specificity,
                                                                    "negative_predictive_value": negative_predictive_value,
                                                                    "f1": f1, "pF1": pF1, "pTPR": pTPR, "pFPR": pFPR})

    allFiles = os.listdir(pathFrom)
    l = pickle.load(open(os.path.join(pathFrom, allFiles[0]), "rb"))

    with tqdm.tqdm(total = l["LCHRMSArea"].shape[0] * len(allFiles), desc="  | .. detecting chromatographic peaks", unit="instances", disable=not verbose) as pbar:

        for fi in os.listdir(pathFrom):
            l = pickle.load(open(os.path.join(pathFrom, fi), "rb"))  ## ['LCHRMSArea', 'areaRTs', 'areaMZs', 'gdProps'])

            lcmsArea = l["LCHRMSArea"]
            assert all(np.amax(lcmsArea, (1,2)) <= 1), "LCHRMSarea is not scaled to a maximum of 1 '%s'"%(str(np.amax(lcmsArea, (1,2))))
            lmProps = l["gdProps"]

            pred = model.predict(lcmsArea)
            peaksDone += lcmsArea.shape[0]

            for j in range(lcmsArea.shape[0]):
                ptyp = np.argmax(np.array(pred[0][j,:]))
                prtInd, pmzInd = [round(i) for i in pred[1][j,:]]
                prtstartInd, prtendInd, pmzstartInd, pmzendInd = [round(i) for i in pred[2][j, :]]

                if ptyp == 0 or ptyp == 1 or ptyp == 2 or ptyp == 3:
                    try:
                        prt, pmz = -1, -1
                        if 0 <= prtInd < l["areaRTs"].shape[1]:
                            prt = l["areaRTs"][j,prtInd]
                        pmz = -1
                        if 0 <= pmzInd < l["areaMZs"].shape[1]:
                            pmz = l["areaMZs"][j,pmzInd]

                        prtstart, prtend, pmzstart, pmzend = -1, -1, -1, -1
                        if 0 <= prtstartInd < l["areaRTs"].shape[1]:
                            prtstart = l["areaRTs"][j,prtstartInd]
                        if 0 <= prtendInd < l["areaRTs"].shape[1]:
                            prtend = l["areaRTs"][j, prtendInd]
                        if 0 <= pmzstartInd < l["areaMZs"].shape[1]:
                            pmzstart = l["areaMZs"][j, pmzstartInd]
                        if 0 <= pmzendInd < l["areaMZs"].shape[1]:
                            pmzend = l["areaMZs"][j, pmzendInd]

                        if "gdProps" in l.keys():
                            gdrt, gdmz, gdrtStartInd, gdrtEndInd, gdmzStartInd, gdmzEndInd = [i for i in l["gdProps"][j,:]]

                            prt = gdrt
                            pmz = gdmz

                        if prt!=-1 and pmz!=-1 and prtstart!=-1 and prtend!=-1 and pmzstart!=-1 and pmzend!=-1:
                            peaks.append([prt, pmz, prtstart, prtend, pmzstart, pmzend, 0, 1])

                    except Exception:
                        errors.append(lmProps[j,0:2])

                elif ptyp == 4:
                    walls.append(lmProps[j,0:2])
                elif ptyp == 5:
                    backgrounds.append(lmProps[j,0:2])
                else:
                    errors.append(lmProps[j,0:2])

            pbar.update(l["LCHRMSArea"].shape[0])

    if verbose:
        print("  | .. %d local maxima analyzed"%(peaksDone))
        print("  | .. of these %7d (%5.1f%%) are chromatographic peaks"%(len(peaks)      , 100 * len(peaks)       / peaksDone))
        print("  | .. of these %7d (%5.1f%%) are backgrounds          "%(len(backgrounds), 100 * len(backgrounds) / peaksDone))
        print("  | .. of these %7d (%5.1f%%) are walls                "%(len(walls)      , 100 * len(walls)       / peaksDone))
        if len(errors) > 0:
            print("  | .. encountered %d errors"%(len(errors)))
        print("  | .. took %.1f seconds"%(toc("detecting with peakbot")))
        print("")

    return peaks, walls, backgrounds, errors

@timeit
def exportPeakBotResultsFeatureML(peaks, fileTo):
    with open(fileTo, "w") as fout:
        fout.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        fout.write('  <featureMap version="1.4" id="fm_16311276685788915066" xsi:noNamespaceSchemaLocation="http://open-ms.sourceforge.net/schemas/FeatureXML_1_4.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n')
        fout.write('    <dataProcessing completion_time="%s">\n'%datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        fout.write('      <software name="PeakBot" version="4.6" />\n')
        fout.write('    </dataProcessing>\n')
        fout.write('    <featureList count="%d">\n'%len(peaks))

        for j, p in enumerate(peaks):
            rt, mz, rtstart, rtend, mzstart, mzend, inte = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            fout.write('<feature id="%s">\n'%j)
            fout.write('  <position dim="0">%f</position>\n'%rt)
            fout.write('  <position dim="1">%f</position>\n'%mz)
            fout.write('  <intensity>%f</intensity>\n'%inte)
            fout.write('  <quality dim="0">0</quality>\n')
            fout.write('  <quality dim="1">0</quality>\n')
            fout.write('  <overallquality>0</overallquality>\n')
            #fout.write('  <charge>1</charge>\n')
            fout.write('  <convexhull nr="0">\n')
            fout.write('    <pt x="%f" y="%f" />\n'%(rtstart , mzstart))
            fout.write('    <pt x="%f" y="%f" />\n'%(rtstart , mzend  ))
            fout.write('    <pt x="%f" y="%f" />\n'%(rtend   , mzend  ))
            fout.write('    <pt x="%f" y="%f" />\n'%(rtend   , mzstart))
            fout.write('  </convexhull>\n')
            fout.write('</feature>\n')

        fout.write('    </featureList>\n')
        fout.write('  </featureMap>\n')
@timeit
def exportPeakBotWallsFeatureML(walls, fileTo):
    with open(fileTo, "w") as fout:
        fout.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        fout.write('  <featureMap version="1.4" id="fm_16311276685788915066" xsi:noNamespaceSchemaLocation="http://open-ms.sourceforge.net/schemas/FeatureXML_1_4.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n')
        fout.write('    <dataProcessing completion_time="%s">\n'%datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        fout.write('      <software name="PeakBot" version="4.6" />\n')
        fout.write('    </dataProcessing>\n')
        fout.write('    <featureList count="%d">\n'%len(walls))

        for j, p in enumerate(walls):
            rt, mz = p[0], p[1]
            fout.write('<feature id="%s">\n'%j)
            fout.write('  <position dim="0">%f</position>\n'%rt)
            fout.write('  <position dim="1">%f</position>\n'%mz)
            fout.write('  <intensity>%f</intensity>\n'%0)
            fout.write('  <quality dim="0">0</quality>\n')
            fout.write('  <quality dim="1">0</quality>\n')
            fout.write('  <overallquality>0</overallquality>\n')
            #fout.write('  <charge>1</charge>\n')
            fout.write('  <convexhull nr="0">\n')
            fout.write('    <pt x="%f" y="%f" />\n'%(rt, mz))
            fout.write('    <pt x="%f" y="%f" />\n'%(rt, mz))
            fout.write('    <pt x="%f" y="%f" />\n'%(rt, mz))
            fout.write('    <pt x="%f" y="%f" />\n'%(rt, mz))
            fout.write('  </convexhull>\n')
            fout.write('</feature>\n')

        fout.write('    </featureList>\n')
        fout.write('  </featureMap>\n')

@timeit
def exportPeakBotResultsTSV(peaks, toFile):
    with open(toFile, "w") as fout:
        fout.write("\t".join(["Num", "RT", "MZ", "RtStart", "RtEnd", "MzStart", "MzEnd", "PeakArea"]))
        fout.write("\n")

        for j, p in enumerate(peaks):
            rt, mz, rtstart, rtend, mzstart, mzend, inte = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            fout.write("\t".join((str(s) for s in [j, rt, mz, rtstart, rtend, mzstart, mzend, inte])))
            fout.write("\n")

@timeit
def exportAreasAsFigures(pathFrom, toFolder, model = None, maxExport = 1E9, threshold=0.01, expFrom = 0, expTo = 1E9):
    cur = 0
    import plotnine as p9
    import pandas as pd

    for fi in tqdm.tqdm(os.listdir(pathFrom)):
        l = pickle.load(open(os.path.join(pathFrom, fi), "rb"))

        lcmsArea = l["LCHRMSArea"]
        areaRTs  = l["areaRTs"]
        areaMZs  = l["areaMZs"]

        pred = None
        if model is not None:
            pred = model.predict(lcmsArea)

        for j in range(lcmsArea.shape[0]):

            if expFrom <= cur <= expTo:

                maxExport = maxExport - 1
                if maxExport < 0:
                    return

                title = None

                p9.options.figure_size = (14,5)

                rts = []
                mzs = []
                ints = []
                alphas = []
                typs = []

                for rowi in range(lcmsArea.shape[1]):
                    for coli in range(lcmsArea.shape[2]):
                        if lcmsArea[j, rowi, coli] > threshold:
                            rts.append(areaRTs[j, rowi])
                            mzs.append(areaMZs[j, coli])
                            ints.append(lcmsArea[j, rowi, coli])
                            typs.append("LCHRMS area (>%f)"%threshold)
                            alphas.append(0.25)
                        if "single" in l.keys() and l["single"][j, rowi, coli] > threshold:
                            rts.append(areaRTs[j, rowi])
                            mzs.append(areaMZs[j, coli])
                            ints.append(l["single"][j, rowi, coli])
                            typs.append("single (>%f)"%threshold)
                            alphas.append(0.25)

                if pred is not None and len(pred)>3:
                    single = pred[3][j]
                    for rowi in range(lcmsArea.shape[1]):
                        for coli in range(lcmsArea.shape[2]):
                            if pred[3][j, rowi, coli, 0]>0.5:
                                rts.append(areaRTs[j, rowi])
                                mzs.append(areaMZs[j, coli])
                                ints.append(1)
                                typs.append("pred - mask")
                                alphas.append(0.25)

                dat = pd.DataFrame({"rts": rts, "mzs": mzs, "Intensity": ints, "typ": typs, "alpha": alphas})
                ## Plot LCMS area and mask
                plot = (p9.ggplot(dat, p9.aes("rts", "mzs", alpha="alpha", colour="Intensity")) +
                        p9.geom_point() +
                        p9.facet_wrap("~typ", ncol=6) +
                        p9.xlim(areaRTs[j,0], areaRTs[j, areaRTs.shape[1]-1]) + p9.ylim(areaMZs[j,0], areaMZs[j, areaMZs.shape[1]-1])
                       )
                ## Plot bounding-box and center
                plot = (plot + p9.scale_alpha(guide = None))

                if "peakType" in l.keys():
                    rt, mz, rtLeftBorder, rtRightBorder, mzLowest, mzHighest = -1, -1, -1, -1, -1, -1
                    typ = int(l["peakType"][j,0])
                    apexRTInd, apexMZInd = [int(i) for i in l["center"][j,:]]
                    rtLeftBorderInd, rtRightBorderInd, mzLowestInd, mzHighestInd = [int(i) for i in l["box"][j,:]]
                    if 0 <= apexRTInd < l["areaRTs"].shape[1]:
                        rt = l["areaRTs"][j,apexRTInd]
                    if 0 <= rtLeftBorderInd < l["areaRTs"].shape[1]:
                        rtLeftBorder = l["areaRTs"][j,rtLeftBorderInd]
                    if 0 <= rtRightBorderInd < l["areaRTs"].shape[1]:
                        rtRightBorder = l["areaRTs"][j,rtRightBorderInd]

                    if 0 <= apexMZInd < l["areaMZs"].shape[1]:
                        mz = l["areaMZs"][j,apexMZInd]
                    if 0 <= mzLowestInd < l["areaMZs"].shape[1]:
                        mzLowest = l["areaMZs"][j,mzLowestInd]
                    if 0 <= mzHighestInd < l["areaMZs"].shape[1]:
                        mzHighest = l["areaMZs"][j,mzHighestInd]

                    plot = (plot + p9.geom_rect(xmin = rtLeftBorder,
                                                xmax = rtRightBorder,
                                                ymin = mzLowest,
                                                ymax = mzHighest,
                                                fill=None,
                                                color="yellowgreen",
                                                linetype="solid",
                                                size=0.1) +
                                   p9.geom_text(label="X",
                                                x=rt,
                                                y=mz,
                                                color="yellowgreen") )

                    title = "" if title is None else title + "\n"
                    title = title + "Ground truth: RT %.2f (box %.2f - %.2f), MZ %.4f (box %.4f - %.4f, %.1f ppm)"%(rt, rtLeftBorder, rtRightBorder, mz, mzLowest, mzHighest, (mzHighest-mzLowest)*1E6/mz)

                if "gdProps" in l.keys():
                    gdrtInd, gdmzInd, gdrtStartInd, gdrtEndInd, gdmzStartInd, gdmzEndInd = [int(i) for i in l["gdProps"][j,:]]
                    gdrt, gdmz, gdrtstart, gdrtend, gdmzstart, gdmzend = -1, -1, -1, -1, -1, -1

                    if 0 <= gdrtInd < l["areaRTs"].shape[1]:
                        gdrt = l["areaRTs"][j,gdrtInd]
                    if 0 <= gdmzInd < l["areaMZs"].shape[1]:
                        gdmz = l["areaMZs"][j,gdmzInd]

                    if 0 <= gdrtStartInd < l["areaRTs"].shape[1]:
                        gdrtstart = l["areaRTs"][j,gdrtStartInd]
                    if 0 <= gdrtEndInd < l["areaRTs"].shape[1]:
                        gdrtend = l["areaRTs"][j, gdrtEndInd]

                    if 0 <= gdmzStartInd < l["areaMZs"].shape[1]:
                        gdmzstart = l["areaMZs"][j, gdmzStartInd]
                    if 0 <= gdmzEndInd < l["areaMZs"].shape[1]:
                        gdmzend = l["areaMZs"][j, gdmzEndInd]

                    plot = (plot + p9.geom_rect(xmin = gdrtstart,
                                                xmax = gdrtend,
                                                ymin = gdmzstart,
                                                ymax = gdmzend,
                                                fill=None,
                                                color="orange",
                                                linetype="solid",
                                                size=0.1) +
                                   p9.geom_text(label="X",
                                                x=gdrt,
                                                y=gdmz,
                                                color="orange"))

                    title = "" if title is None else title + "\n"
                    title = title + "Local maxima: RT %.2f (box %.2f - %.2f), MZ %.4f (box %.4f - %.4f, %.1f ppm)"%(gdrt, gdrtstart, gdrtend, gdmz, gdmzstart, gdmzend, (gdmzend-gdmzstart)*1E6/gdmz)

                prt, pmz = -1, -1
                prtstart, prtend, pmzstart, pmzend = -1, -1, -1, -1
                if pred is not None:
                    pisFullPeak, pisPartPeak, phasCoElutionLeft, phasCoElutionRight, pisBackground, pisWall = [bool(i) for i in np.squeeze(np.eye(6)[np.argmax(np.array(pred[0][j,:])).reshape(-1)])]
                    prtInd, pmzInd = [round(i) for i in pred[1][j,:]]
                    prtstartInd, prtendInd, pmzstartInd, pmzendInd = [round(i) for i in pred[2][j, :]]

                    if pisFullPeak or pisPartPeak or phasCoElutionLeft or phasCoElutionRight:
                        try:
                            ## properties: rt, mz, rtFirstSlice, rtLastSlice, mzFirstSlice, mzLastSlice, ppmdev, leftRTBorder, rightRTBorder
                            if 0 <= prtInd < l["areaRTs"].shape[1]:
                                prt = l["areaRTs"][j,prtInd]
                            if 0 <= pmzInd < l["areaMZs"].shape[1]:
                                pmz = l["areaMZs"][j,pmzInd]

                            if 0 <= prtstartInd < l["areaRTs"].shape[1]:
                                prtstart = l["areaRTs"][j,prtstartInd]
                            if 0 <= prtendInd < l["areaRTs"].shape[1]:
                                prtend = l["areaRTs"][j, prtendInd]

                            if 0 <= pmzstartInd < l["areaMZs"].shape[1]:
                                pmzstart = l["areaMZs"][j, pmzstartInd]
                            if 0 <= pmzendInd < l["areaMZs"].shape[1]:
                                pmzend = l["areaMZs"][j, pmzendInd]

                            plot = (plot + p9.geom_rect(xmin = prtstart,
                                                        xmax = prtend,
                                                        ymin = pmzstart,
                                                        ymax = pmzend,
                                                        fill=None,
                                                        color="firebrick",
                                                        linetype="solid",
                                                        size=0.1) +
                                           p9.geom_text(label="X",
                                                        x=prt,
                                                        y=pmz,
                                                        color="firebrick")) # + p9.geom_hline(yintercept=pmz)+p9.geom_vline(xintercept=prt))
                        except Exception:
                            import traceback
                            traceback.print_exc()

                    title = "" if title is None else title + "\n"
                    title = title + "Predicted   : RT %.2f (box %.2f - %.2f), MZ %.4f (box %.4f - %.4f, %.1f ppm); Type '%s'"%(prt, prtstart, prtend, pmz, pmzstart, pmzend, (pmzend-pmzstart)*1E6/pmz, ["Single peak", "Isomers earlier and later", "Isomer earlier", "Isomer later", "Background", "Wall"][np.argmax(np.array(pred[0][j,:]))])
                plot = (plot +
                        p9.ggtitle(title) + p9.xlab("Retention time (seconds)") + p9.ylab("m/z"))

                p9.ggsave(plot=plot, filename=os.path.join(toFolder, "%d.png"%(cur)), height=7, width=12)
            cur += 1




            
            
            
            
            
            
            


@timeit
def exportGroupedFeaturesAsFeatureML(headers, features, fileTo):
    with open(fileTo, "w") as fout:
        fout.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        fout.write('  <featureMap version="1.4" id="fm_16311276685788915066" xsi:noNamespaceSchemaLocation="http://open-ms.sourceforge.net/schemas/FeatureXML_1_4.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n')
        fout.write('    <dataProcessing completion_time="%s">\n'%datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        fout.write('      <software name="PeakBot" version="4.6" />\n')
        fout.write('    </dataProcessing>\n')
        fout.write('    <featureList count="%d">\n'%len(features))
        
        
        for j, feature in enumerate(features):
            rt, mz, rtstart, rtend, mzstart, mzend = feature[0:6]
            inte = 1
            fout.write('<feature id="%s">\n'%j)
            fout.write('  <position dim="0">%f</position>\n'%rt)
            fout.write('  <position dim="1">%f</position>\n'%mz)
            fout.write('  <intensity>%f</intensity>\n'%inte)
            fout.write('  <quality dim="0">0</quality>\n')
            fout.write('  <quality dim="1">0</quality>\n')
            fout.write('  <overallquality>0</overallquality>\n')
            #fout.write('  <charge>1</charge>\n')
            fout.write('  <convexhull nr="0">\n')
            fout.write('    <pt x="%f" y="%f" />\n'%(rtstart , mzstart))
            fout.write('    <pt x="%f" y="%f" />\n'%(rtstart , mzend  ))
            fout.write('    <pt x="%f" y="%f" />\n'%(rtend   , mzend  ))
            fout.write('    <pt x="%f" y="%f" />\n'%(rtend   , mzstart))
            fout.write('  </convexhull>\n')
            fout.write('</feature>\n')

        fout.write('    </featureList>\n')
        fout.write('  </featureMap>\n')

@timeit
def exportGroupedFeaturesAsTSV(headers, features, toFile):
    writeTSVFile(toFile, headers, features)





















@timeit
def exportLocalMaximaAsFeatureML(foutFile, peaks):
    with open(foutFile, "w") as fout:
        fout.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        fout.write('  <featureMap version="1.4" id="fm_16311276685788915066" xsi:noNamespaceSchemaLocation="http://open-ms.sourceforge.net/schemas/FeatureXML_1_4.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n')
        fout.write('    <dataProcessing completion_time="%s">\n'%datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        fout.write('      <software name="PeakBot" version="4.5" />\n')
        fout.write('    </dataProcessing>\n')
        fout.write('    <featureList count="%d">\n'%peaks.shape[0])

        for j in range(peaks.shape[0]):
            rt = peaks[j,2]
            mz = peaks[j,3]
            inte = peaks[j, 4]
            fout.write('<feature id="%s">\n'%j)
            fout.write('  <position dim="0">%f</position>\n'%rt)
            fout.write('  <position dim="1">%f</position>\n'%mz)
            fout.write('  <intensity>%f</intensity>\n'%inte)
            fout.write('  <quality dim="0">0</quality>\n')
            fout.write('  <quality dim="1">0</quality>\n')
            fout.write('  <overallquality>0</overallquality>\n')
            fout.write('  <charge>1</charge>\n')
            fout.write('  <convexhull nr="0">\n')
            fout.write('    <pt x="%f" y="%f" />\n'%(peaks[j,8] , peaks[j, 3]*(1.-peaks[j,5]/6/1.E6)))
            fout.write('    <pt x="%f" y="%f" />\n'%(peaks[j,8] , peaks[j, 3]*(1.+peaks[j,5]/6/1.E6)))
            fout.write('    <pt x="%f" y="%f" />\n'%(peaks[j,11], peaks[j, 3]*(1.+peaks[j,5]/6/1.E6)))
            fout.write('    <pt x="%f" y="%f" />\n'%(peaks[j,11], peaks[j, 3]*(1.-peaks[j,5]/6/1.E6)))
            fout.write('  </convexhull>\n')
            fout.write('</feature>\n')

        fout.write('    </featureList>\n')
        fout.write('  </featureMap>\n')
@timeit
def exportLocalMaximaAsTSV(foutFile, peaks):
    with open(foutFile, "w") as fout:
        fout.write("\t".join(["Num", "RT", "MZ", "RtStart", "RtEnd", "MzDeviation"]))
        fout.write("\n")

        for j in range(peaks.shape[0]):
            fout.write("\t".join((str(s) for s in [j, peaks[j,2], peaks[j,3], peaks[j,8], peaks[j,11], peaks[j,5]])))
            fout.write("\n")
def showSummaryPlots(peaks, maximaProps, maximaPropsAll, polarity, highlights = None):
    tic()
    print("Feature map of all local maxima (grey) and local maxima with peak like shapes")
    plt.scatter(maximaPropsAll[:,2], maximaPropsAll[:,3], color="slategrey")
    plt.scatter(peaks[:,2], peaks[:,3], color="Firebrick")
    if polarity == "positive":
        pheInd = np.argwhere(np.logical_and(np.abs(maximaProps[:,2] - 290) < 10, np.abs(maximaProps[:,3] - 166.08644) < 0.015))
        plt.scatter(maximaProps[pheInd,2], maximaProps[pheInd,3], color = "Orange")
    plt.title("Feature map"); plt.xlabel("Rt (seconds)"); plt.ylabel("M/Z")
    plt.show(block = False)

    if highlights != None and len(highlights) > 0:
        ## highlights must be a dict of "name": (rt in sec, mz, rt plus/minus, ppm plus/minus)

        for name, (rt, mz, rtpm, mzpm) in highlights.items():
            print("\n\n\n%s features"%name)
            plt.scatter(maximaPropsAll[:,2], maximaPropsAll[:,3], color="slategrey")
            plt.scatter(peaks[:,2], peaks[:,3], color="Firebrick")
            print("peaks")
            pheInd = np.argwhere(np.logical_and(np.abs(peaks[:,2] - rt) < rtpm, np.abs(peaks[:,3] - mz)*1E6/mz < mzpm))
            print(peaks[pheInd,])
            plt.scatter(peaks[pheInd,2], peaks[pheInd,3], color = "Orange")
            print("all maxima")
            pheInd = np.argwhere(np.logical_and(np.abs(maximaPropsAll[:,2] - rt) < rtpm, np.abs(maximaPropsAll[:,3] - mz)*1E6/mz < mzpm))
            print(maximaPropsAll[pheInd,])
            plt.title("Feature map (Phe area)"); plt.xlabel("Rt (seconds)"); plt.ylabel("M/Z")
            plt.xlim([rt-rtpm, rt+rtpm]); plt.ylim([mz*(1-mzpm/1E6), mz*(1+mzpm/1E6)])
            plt.show()

    print("\n\n\n")
    def funcLogarithmic(x,a,b,c):
        return a * np.sqrt(x - b) + c
    popt, pcov = scipy.optimize.curve_fit(funcLogarithmic, peaks[:,3], peaks[:,5])
    plt.scatter(peaks[:,3], peaks[:,5], alpha=0.02)
    res = scipy.stats.linregress(peaks[:,3], peaks[:,5])
    plt.plot(np.sort(peaks[:,3]), res.intercept + res.slope*np.sort(peaks[:,3]), 'r')
    plt.plot(np.sort(peaks[:,3]), funcLogarithmic(np.sort(peaks[:,3]), *popt), 'r--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    diffH13C = (2.01410177812 - 1.00782503223) - (13.00335483507 - 12)
    plt.plot(range(100, 1001, 10), [diffH13C * 1E6 / i for i in range(100, 1001, 10)], 'red')
    diff34S13C2 = 2*(13.00335483507 - 12) - (33.967867004 - 31.9720711744)
    plt.plot(range(100, 1001, 10), [diff34S13C2 * 1E6 / i for i in range(100, 1001, 10)], 'yellow')
    diff15N13C = (13.00335483507 - 12) - (15.00010889888 - 14.00307400443)
    plt.plot(range(100, 1001, 10), [diff15N13C * 1E6 / i for i in range(100, 1001, 10)], 'blue')
    plt.ylim([0, 60])
    plt.title("Peak with in m/z dimension"); plt.xlabel("m/z"); plt.ylabel("profile mode width (ppm)")
    print("Sqrt-regression of peak-width in the m/z dimension: ppm dev at mz 100: %.4f; ppm dev at mz 900: %.4f"%(funcLogarithmic(100, *popt), funcLogarithmic(900, *popt)))
    plt.show(block = False)

    print("\n\n\n")
    temp = peaks[:,6] *1E6 / peaks[:,3]
    popt, pcov = scipy.optimize.curve_fit(funcLogarithmic, peaks[:,3], temp)
    plt.scatter(peaks[:,3], temp, alpha=0.02)
    res = scipy.stats.linregress(peaks[:,3], temp)
    plt.plot(np.sort(peaks[:,3]), res.intercept + res.slope*np.sort(peaks[:,3]), 'r')
    plt.plot(np.sort(peaks[:,3]), funcLogarithmic(np.sort(peaks[:,3]), *popt), 'r--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.title("MZ profile peak adjacent signals difference"); plt.xlabel("M/Z"); plt.ylabel("M/Z difference (ppm)")
    print("Sqrt-regression of difference in adjacent signals in the m/z dimension")
    plt.show(block = False)

    print("\n\n\n")
    print("Peak width in the RT dimension (start of peak (border) to peak apex)")
    plt.hist(peaks[:, 2] - peaks[:, 8], bins=range(0, 100, 1))
    plt.title("RT left border (rel to apex)"); plt.xlabel("Offset (scans)"); plt.ylabel("Count")
    plt.show(block = False)
    plt.hist(peaks[:, 11] - peaks[:, 2], bins=range(0, 100, 1))
    plt.title("RT right border (rel to apex)"); plt.xlabel("Offset (scans)"); plt.ylabel("Count")
    plt.show(block = False)

    tocAddStat("Summary overview and graphs")

    
    
def _findMZGeneric(mzs, ints, times, peaksCount, scanInd, mzRef, mzleft, mzright):

    if peaksCount[scanInd]==0:
        return -1

    min = 0
    max = peaksCount[scanInd]

    while min <= max:
        cur = int((max + min) // 2)

        if mzleft <= mzs[scanInd, cur] <= mzright:
            
            ## continue search to the left side
            while cur-1 >= 0 and mzs[scanInd, cur-1] >= mzleft and abs(mzRef - mzs[scanInd, cur-1]) < abs(mzRef - mzs[scanInd, cur]):
                cur = cur - 1
            ## continue search to the right side
            while cur+1 < peaksCount[scanInd] and mzs[scanInd, cur+1] <= mzright and abs(mzRef - mzs[scanInd, cur+1]) < abs(mzRef - mzs[scanInd, cur]):
                cur = cur + 1
            
            return cur

        if mzs[scanInd, cur] > mzright:
            max = cur - 1
        else:
            min = cur + 1

    return -1
    
def estimateParameters(chrom, to):
    mzs, ints, times, peaksCount = chrom.convertChromatogramToNumpyObjects(verbose = False)
    
    ## calculate differences intraScan
    mz = mzs[:,1:].flatten()
    diff = np.diff(mzs).flatten()*1E6/mz
    df = pd.DataFrame({"MZ": mz, "DiffPPM": diff})
    
    ## remove non-signals
    df = df.where(df["MZ"]<100000000.0)
    
    
    ## select largest MZ values
    maxmz = df["MZ"].max()
    minmz = df["MZ"].min()
    incMZ = minmz + (maxmz - minmz) * 0.9
    df = df.where(df["MZ"] >= incMZ)
        
    df.sort_values("MZ", inplace=True, ascending=False)
    df.dropna(inplace = True)
    
    minV = df["DiffPPM"].min()
    medianV = df["DiffPPM"].median()
    meanV = df["DiffPPM"].mean()
    maxV = df["DiffPPM"].max()
    print("Intra-scan PPM differences are min: %.1f, median: %.1f, mean: %.1f, max: %.1f"%(minV, medianV, meanV, maxV))
    
    suggestedIntraScanPPM = minV * 4
    
    mz = []
    diff = []
    for scanInd in range(mzs.shape[0]):
        for peakInd in range(peaksCount[scanInd]):
            if scanInd > 0 and mzs[scanInd, peakInd] >= incMZ:
                pI = _findMZGeneric(mzs, ints, times, peaksCount, scanInd-1, mzs[scanInd, peakInd], mzs[scanInd, peakInd]*(1-suggestedIntraScanPPM/1E6), mzs[scanInd, peakInd]*(1+suggestedIntraScanPPM/1E6))
                if pI > -1:
                    oMZ = mzs[scanInd-1, pI]
                    mz.append(mzs[scanInd, peakInd])
                    diff.append(abs(oMZ-mzs[scanInd, peakInd])*1E6/mzs[scanInd, peakInd])
    
    df = pd.DataFrame({"MZ": mz, "DiffPPM": diff})
        
    df.sort_values("DiffPPM", inplace=True, ascending=False)
    
    minV = df["DiffPPM"].min()
    medianV = df["DiffPPM"].median()
    meanV = df["DiffPPM"].mean()
    maxV = df["DiffPPM"].max()
    print("Inter-scan PPM differences are min: %.1f, median: %.1f, mean: %.1f, max: %.1f"%(minV, medianV, meanV, maxV))
    
    return suggestedIntraScanPPM
    








@numba.jit(nopython=True, parallel=True)
def _expMSMSExcList(temp, peaks, ppm, rtAdd):
    for ti in numba.prange(temp.shape[0]):
        useAsExclusion = True
        useAsWallExclusion = True
        for pi in range(peaks.shape[0]):
            if abs(temp[ti, 1]-peaks[pi, 1])*1E6/peaks[pi, 1] <= ppm:
                useAsWallExclusion = False
                if peaks[pi, 2]-rtAdd <= temp[ti, 0] <= peaks[pi, 3]+rtAdd:
                    useAsExclusion = False
                
        temp[ti, 2] = useAsExclusion
        temp[ti, 4] = temp[ti, 0] - rtAdd
        temp[ti, 5] = temp[ti, 0] + rtAdd
        temp[ti, 3] = useAsWallExclusion

@numba.jit(nopython=True)
def _expMSMSExcList2(temp, peaks, ppm, rtAdd, rtStart, rtEnd):
    for ti in range(temp.shape[0]):
        if temp[ti,3] != 0:
            for pi in range(temp.shape[0]):
                if temp[pi,3]!=0 and abs(temp[ti, 1]-temp[pi, 1])*1E6/temp[pi, 1] <= ppm:
                    temp[pi,3] = 0
@numba.jit(nopython=True)
def _expMSMSExcList3(temp, peaks, ppm, rtAdd, rtStart, rtEnd):
    for ti in range(temp.shape[0]):
        if temp[ti,2] != 0:
            for pi in range(temp.shape[0]):
                if ti!=pi and temp[pi,2]!=0 and abs(temp[ti, 1]-temp[pi, 1])*1E6/temp[pi, 1] <= ppm:
                    if temp[ti,4] <= temp[pi,4] and temp[pi,5] <= temp[ti,5]:
                        temp[pi,2] = 0
                    elif temp[ti,4] <= temp[pi,4] <= temp[ti,5] <= temp[pi,5]:
                        temp[pi,2] = 0
                        temp[ti,5] = temp[pi, 5]
                    elif temp[pi, 4] <= temp[ti,4] <= temp[pi,5] <= temp[ti,5]:
                        temp[pi,2] = 0
                        temp[ti, 4] = temp[pi, 4]

def exportMSMSExclusionList(walls, backgrounds, peaks, fileTo, ppm = 5., rtAdd = 30., rtStart=100, rtEnd=2400, verbose = True):
    tic("startMSMSExcListFunction")
    if verbose: 
        print("Generating MSMS exclusion list")

    temp = np.zeros((len(walls)+len(backgrounds), 6))  ## rt, mz, uselocal, useGlobal, startrt, endrt
    cur = 0
    for i in walls:
        temp[cur,0] = i[0]
        temp[cur,1] = i[1]
        cur = cur + 1
    for i in backgrounds:
        temp[cur,0] = i[0]
        temp[cur,1] = i[1]
        cur = cur + 1

    peaks = np.array(peaks)
    _expMSMSExcList(temp, peaks, ppm, rtAdd)
    temp = temp[np.logical_or(temp[:,2]!=0, temp[:,3]!=0),:]
    print("There are %d exclustions"%temp.shape[0])
    _expMSMSExcList2(temp, peaks, ppm, rtAdd, rtStart, rtEnd)
    temp = temp[np.logical_or(temp[:,2]!=0, temp[:,3]!=0),:]
    print("There are %d exclustions"%temp.shape[0])

    
    for i in range(3):
        if verbose: print("  | .. iteration %d"%i)
        _expMSMSExcList3(temp, peaks, ppm, rtAdd, rtStart, rtEnd)
        temp = temp[np.logical_or(temp[:,2]!=0, temp[:,3]!=0),:]
        print("There are %d exclustions"%temp.shape[0])
    
    temp = temp[(temp[:,5]-temp[:,4])>180,:]

    ex = 0
    with open(fileTo, "w") as fout:
        fout.write("MZ\tRTStart\tRTEnd\n")
        for ti in range(temp.shape[0]):
            if temp[ti, 3] != 0:
                fout.write("%f\t%f\t%f\n"%(temp[ti, 1], rtStart, rtEnd))
            else:
                fout.write("%f\t%f\t%f\n"%(temp[ti, 1], temp[ti, 4], temp[ti, 5]))
            ex = ex + 1

    with open(fileTo+".featureML", "w") as fout:
        fout.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        fout.write('  <featureMap version="1.4" id="fm_16311276685788915066" xsi:noNamespaceSchemaLocation="http://open-ms.sourceforge.net/schemas/FeatureXML_1_4.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n')
        fout.write('    <dataProcessing completion_time="%s">\n'%datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        fout.write('      <software name="PeakBot" version="4.5" />\n')
        fout.write('    </dataProcessing>\n')
        fout.write('    <featureList count="%d">\n'%(ex))

        for ti in range(temp.shape[0]):
            rt = temp[ti,0]
            mz = temp[ti,1]
            fout.write('<feature id="%s">\n'%ti)
            fout.write('  <position dim="0">%f</position>\n'%rt)
            fout.write('  <position dim="1">%f</position>\n'%mz)
            fout.write('  <intensity>%f</intensity>\n'%0)
            fout.write('  <quality dim="0">0</quality>\n')
            fout.write('  <quality dim="1">0</quality>\n')
            fout.write('  <overallquality>0</overallquality>\n')
            fout.write('  <charge>1</charge>\n')
            fout.write('  <convexhull nr="0">\n')
            if temp[ti, 3] != 0:
                fout.write('    <pt x="%f" y="%f" />\n'%(rtStart, mz*(1.-ppm/1.E6)))
                fout.write('    <pt x="%f" y="%f" />\n'%(rtStart, mz*(1.+ppm/1.E6)))
                fout.write('    <pt x="%f" y="%f" />\n'%(rtEnd, mz*(1.+ppm/1.E6)))
                fout.write('    <pt x="%f" y="%f" />\n'%(rtEnd, mz*(1.-ppm/1.E6)))
            else:
                fout.write('    <pt x="%f" y="%f" />\n'%(temp[ti, 4], mz*(1.-ppm/1.E6)))
                fout.write('    <pt x="%f" y="%f" />\n'%(temp[ti, 4], mz*(1.+ppm/1.E6)))
                fout.write('    <pt x="%f" y="%f" />\n'%(temp[ti, 5], mz*(1.+ppm/1.E6)))
                fout.write('    <pt x="%f" y="%f" />\n'%(temp[ti, 5], mz*(1.-ppm/1.E6)))
            fout.write('  </convexhull>\n')
            fout.write('</feature>\n')

        fout.write('    </featureList>\n')
        fout.write('  </featureMap>\n')
    
    if verbose: print("  | .. took %.1f seconds"%(toc("startMSMSExcListFunction")))
    

    
            
