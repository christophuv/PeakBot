
from .core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary

import math
import os
import datetime
import pickle
import multiprocessing
import uuid

import tensorflow as tf
import numpy as np
import scipy
import pandas as pd

import tqdm




#####################################
### Configuration class
##
class Config(object):
    """Base configuration class"""

    NAME    = "PeakBot"
    VERSION = "4.6"
    
    RTSLICES   =  32   ## should be of 2^n
    MZSLICES   = 128   ## should be of 2^m    
    NUMCLASSES =   6   ## [isFullPeak, hasCoelutingPeakLeftAndRight, hasCoelutingPeakLeft, hasCoelutingPeakRight, isWall, isBackground]
    
    BATCHSIZE     = 128
    STEPSPEREPOCH =  64
    EPOCHS        = 512
    
    DROPOUT        = 0.2
    UNETLAYERSIZES = [32, 64, 128, 256]
    
    LEARNINGRATESTART              = 0.005
    LEARNINGRATEDECREASEAFTERSTEPS = 5
    LEARNINGRATEMULTIPLIER         = 0.7
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
    

    
    
    
    
    
    
    
    

#####################################
### Data generator methods
### Read files from a directory and prepare them
### for PeakBot training and prediction
##
def dataGenerator(folder, instancePrefix = None, verbose=False):
    
    if instancePrefix is None:
        instancePrefix = Config.INSTANCEPREFIX
    
    ite = 0
    while os.path.isfile("%s/%s%d.pickle"%(folder, instancePrefix, ite)):
            
        l = pickle.load(open("%s/%s%d.pickle"%(folder, instancePrefix, ite), "rb"))
        
        l["LCHRMSArea"] = np.expand_dims(l["LCHRMSArea"], len(l["LCHRMSArea"].shape))
        for j in range(l["LCHRMSArea"].shape[0]):
            m = np.max(l["LCHRMSArea"][j,:,:,:])
            if m > 0:
                l["LCHRMSArea"][j,:,:,:] = l["LCHRMSArea"][j,:,:,:] / m
        
        yield {"LCHRMSArea"      : l["LCHRMSArea"      ],
               "peakType"        : l["peakType"        ],
               "areaRTs"         : l["areaRTs"         ], 
               "areaMZs"         : l["areaMZs"         ], 
               "center"          : l["center"          ], 
               "box"             : l["box"             ], 
               "chromatogramFile": l["chromatogramFile"]}
        
        ite += 1
        
def modelAdapterTrainGenerator(datGen, newBatchSize = None, verbose=False):
    ite = 0
    l = next(datGen)
    while l is not None:
        
        if verbose and ite == 0:
            print("  | Generated data is")

            for k, v in l.items():
                if type(v).__module__ == "numpy":
                    print("  | .. gt: %18s numpy: %30s %10s"%(k, v.shape, v.dtype))
                else:
                    print("  | .. gt: %18s  type:  %40s"%(k, type(v)))
            print("")
        
        if newBatchSize is not None:
            
            for k in l.keys():
                if isinstance(l[k], np.ndarray) and len(l[k].shape)==2:
                    l[k] = l[k][0:newBatchSize,:]

                elif isinstance(l[k], np.ndarray) and len(l[k].shape)==3:
                    l[k] = l[k][0:newBatchSize,:,:]

                elif isinstance(l[k], np.ndarray) and len(l[k].shape)==4:
                    l[k] = l[k][0:newBatchSize,:,:,:]

                elif isinstance(l[k], list):
                    l[k] = l[k][0:newBatchSize]
        
        yield {"LCHRMSArea": l["LCHRMSArea"]}, {"peakType"        : l["peakType"        ], 
                                                "center"          : l["center"          ], 
                                                "box"             : l["box"             ]}
        ite += 1   
        
def modelAdapterTestGenerator(datGen, newBatchSize = None, verbose=False):
    return modelAdapterTrainGenerator(datGen, newBatchSize, verbose)



    
    
    
    
    
    
    
    
    


#####################################
### PeakBot additional methods
##
def batch_iou(boxes1, boxes2):
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
    def __init__(self, validation_sets, verbose=0, batch_size=None, steps=None, everyNthEpoch=1):
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
    
    def addValidationSet(validation_set):
        if len(validation_set) not in [3, 4]:
            raise ValueError()
        self.validation_sets.append(validation_set)
        
    
    @timeit
    def on_epoch_end(self, epoch, logs=None, ignoreEpoch = False):
        self.lastEpochNum = epoch
        hist = None
        if epoch%self.everyNthEpoch == 0 or ignoreEpoch:
            hist={}
            if self.verbose: print("  | .. on_epoch_end: Additional test datasets (epoch %d): "%(epoch+1), end="")
            add = False
            # evaluate on the additional validation sets
            for validation_set in self.validation_sets:
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
                
                file_writer = tf.summary.create_file_writer(logDir + "/" + validation_set_name)
                for metric, result in zip(self.model.metrics_names,results):
                    valuename = "epoch_" + metric
                    with file_writer.as_default():
                        tf.summary.scalar(valuename, data=result, step=epoch)
                    if add:
                        if self.verbose: print(" - ", end="")
                    add=True
                    valuename = validation_set_name + "_" + metric
                    if self.verbose: print("%s: %.4f"%(valuename, result), end="")
                    hist[valuename] = result
            if self.verbose: print("")
        self.history.append(hist)
                
    def on_train_end(self, logs=None):
        self.on_epoch_end(self.lastEpochNum, logs=logs, ignoreEpoch = True)
    
class PeakBot():
    def __init__(self, name, batchSize = None, rts = None, mzs = None, numClasses = None, version = None):
        if batchSize is None:
            batchSize = Config.BATCHSIZE
        if rts is None:
            rts = Config.RTS
        if mzs is None:
            mzs = Config.MZS
        if numClasses is None:
            numClasses = Config.NUMCLASSES
        if version is None:
            version = Config.VERSION
            
        self.name       = name
        self.batchSize  = batchSize
        self.rts        = rts
        self.mzs        = mzs
        self.numClasses = numClasses        
        self.model      = None        
        self.version    = version
            
    
    @timeit
    def buildTFModel(self, verbose = False, dropOutRate = None, uNetLayerSizes = None):
        if dropOutRate is None:
            dropOutRate = Config.DROPOUT
        if uNetLayerSizes is None:
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
            
            x = tf.keras.layers.Dropout(dropOutRate)(x)
            x = tf.keras.layers.ZeroPadding2D(padding=1)(x)
            x = tf.keras.layers.Conv2D(uNetLayerSizes[i], (3,3), use_bias=True)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            
            x = tf.keras.layers.MaxPool2D((2,2))(x)
            cLayers.append(x)
        lastUpLayer = x
        
        ## Intermediate layer and feature properties (indices and borders)
        fx          = tf.keras.layers.Flatten()(x)
        peakType    = tf.keras.layers.Dense(self.numClasses, name="peakType", activation="sigmoid")(fx)
        center      = tf.keras.layers.Dense(              2, name="center"                        )(fx)
        box         = tf.keras.layers.Dense(              4, name="box"                           )(fx)
      
        ## Decoder
        for i in range(len(uNetLayerSizes)-1, -1, -1):
            x = tf.keras.layers.UpSampling2D((2,2))(x)
            x = tf.concat([x, cLayers[i]], axis=3)
            x = tf.keras.layers.ZeroPadding2D(padding=2)(x) 
            x = tf.keras.layers.Conv2D(uNetLayerSizes[i-1] if i > 0 else 1, (5,5), use_bias=True)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            
            x = tf.keras.layers.Dropout(dropOutRate)(x)
            x = tf.keras.layers.ZeroPadding2D(padding=1)(x) 
            x = tf.keras.layers.Conv2D(uNetLayerSizes[i-1] if i > 0 else 1, (3,3), use_bias=True)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            if i > 0:
                x = tf.keras.layers.Add()([x, cLayers[i]])
            x = tf.keras.layers.Activation("relu")(x)
                
        x = tf.keras.layers.ZeroPadding2D(padding=1)(x)
        x = tf.keras.layers.Conv2D(1, (3,3))(x)
        
        x = tf.keras.layers.Activation("sigmoid", name="mask")(x)
        mask = x
            
        if verbose: 
            print("  | .. Intermediate layer")
            print("  | .. .. lastUpLayer is", lastUpLayer)
            print("  | .. .. fx          is", fx)
            print("  | .. ")            
            print("  | .. Outputs")
            print("  | .. .. peak        is", peakType)
            print("  | .. .. center      is", center)
            print("  | .. .. box         is", box)
            print("  | .. .. mask        is", mask)
            print("  | .. ")
            print("  | ")

        self.model = tf.keras.models.Model(input_, [peakType, center, box])#, mask])
    
    @timeit
    def compileModel(self, learningRate = Config.LEARNINGRATESTART):
        cce = tf.keras.losses.CategoricalCrossentropy()
        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate),
            loss         = {
                        "peakType"   : "CategoricalCrossentropy",
                        "center"     : tf.keras.losses.Huber(),
                        "box"        : tf.keras.losses.Huber(),
                        #"mask"       : "BinaryCrossentropy",
                      },
            metrics      = {
                        "peakType" : ["categorical_accuracy"],
                        "center"   : [],
                        "box"      : [batch_iou],
                        #"mask"     : [],
                      },
            loss_weights = {
                        "peakType"   : 32,
                        "center"     : 32,
                        "box"        : 32,
                        #"mask"       : 1,
                      }
        )
    
    @timeit
    def train(self, datTrain, datVal, epochs = None, steps_per_epoch = None, logDir = None, callbacks = None, verbose = 1):
        if epochs is None: 
            epochs = Config.EPOCHS
        if steps_per_epoch is None:
            steps_per_epoch = Config.STEPSPEREPOCH
        
        if verbose: 
            print("  | Fitting model on training data")
            print("  | .. Logdir is '%s'"%logDir)
            print("  | .. Number of epochs %d"%(epochs))
            print("  |")
        
        if logDir is None:
            logDir = "logs/fit/PeakBot_v" + self.version + "_" + uuid.uuid4().hex
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logDir, histogram_freq = 1, write_graph = True)
        file_writer = tf.summary.create_file_writer(logDir + "/scalars")
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
        self.model = tf.keras.models.load_model(modelFile, custom_objects = {"batch_iou": batch_iou})
    
    def saveModelToFile(self, modelFile):
        self.model.save(modelFile)
        


    
    
    
    
@timeit
def trainPeakBotModel(trainInstancesPath, logBaseDir, modelName = None, valInstancesPath = None, addValidationInstances = None, everyNthEpoch = 15, verbose = False):
    tic("pbTrainNewModel")

    if modelName is None:
        modelName = "%s%s__%s"%(Config.NAME, Config.VERSION, uuid.uuid4().hex[0:6])
        
    logDir = logBaseDir + "/" + modelName
    logger = tf.keras.callbacks.CSVLogger('%s/clog.tsv'%(logDir), separator="\t")
    
    if verbose:
        print("Training new PeakBot model")
        print("  | Model name is '%s'"%(modelName))
        print("  | .. config is")
        print(Config.getAsStringFancy().replace(";", "\n"))
        print("  |")            

    def lrSchedule(epoch, lr):
        if (epoch + 1) % Config.LEARNINGRATEDECREASEAFTERSTEPS == 0:
            lr *= Config.LEARNINGRATEMULTIPLIER
        tf.summary.scalar('learningRate', data=lr, step=epoch)

        return max(lr, Config.LEARNINGRATEMINVALUE)
    lrScheduler = tf.keras.callbacks.LearningRateScheduler(lrSchedule, verbose=False)

    datGenTrain = modelAdapterTrainGenerator(dataGenerator(trainInstancesPath, verbose = verbose), newBatchSize = Config.BATCHSIZE, verbose = verbose)
    datGenVal   = None
    if valInstancesPath is not None:
        datGenVal = modelAdapterTestGenerator(dataGenerator(valInstancesPath  , verbose = verbose), newBatchSize = Config.BATCHSIZE, verbose = verbose)
        datGenVal = tf.data.Dataset.from_tensors(next(datGenVal))

    valDS = AdditionalValidationSets(None, 
                                     batch_size=Config.BATCHSIZE,
                                     everyNthEpoch = everyNthEpoch, 
                                     verbose = verbose)
    if addValidationInstances is not None:
        for valInstance in addValidationInstances:
            datGen  = modelAdapterTrainGen(dataGenerator(valInstance["folder"]), 
                                         newBatchSize = Config.BATCHSIZE)
            x,y = convertGeneratorToPlain(datGen, 1)
            valDS.addValidationSet((x,y, valInstance["name"]))
        
        
    pb = PeakBot(modelName, batchSize = Config.BATCHSIZE, rts = Config.RTSLICES, mzs = Config.MZSLICES)
    pb.buildTFModel(dropOutRate = Config.DROPOUT, uNetLayerSizes=Config.UNETLAYERSIZES, verbose = verbose)
    pb.compileModel(learningRate = Config.LEARNINGRATESTART)
    
    history = pb.train(
        datTrain = datGenTrain, 
        datVal   = datGenVal, 

        epochs          = Config.EPOCHS,
        steps_per_epoch = Config.STEPSPEREPOCH,

        logDir = logDir, 
        callbacks = [logger, lrScheduler, valDS],

        verbose = verbose * 2
    )
    
    
    metricesAddValDS = pd.DataFrame(columns=["model", "set", "metric", "value"]);
    if addValidationInstances is not None:
        hist = addValDatasets.history[-1]
        for valInstance in addValidationInstances:
    
            se = valInstance["name"]
            for metric in ["loss", "peakType_loss", "center_loss", "box_loss", "mask_loss", "peakType_categorical_accuracy", "box_batch_iou"]:
                val = hist[se + "_" + metric]
                newRow = pd.Series({"model": modelName, "set": se, "metric": metric, "value": val})
                metricesAddValDS = metricesAddValDS.append(newRow, ignore_index=True)
    
    if verbose: 
        print("  |")
        print("  | .. model built and trained successfully (took %.1f seconds)"%toc("pbTrainNewModel"))
    
    return pb, metricesAddValDS


        
        
        
        
        
        
        
        
        
        
        
        
        
        
@timeit
def runPeakBot(pathFrom, modelPath, verbose = True):
    tic("running peakbot")
    
    peaks = []
    backgrounds = 0
    walls = 0
    errors = 0

    peaksDone = 0
    debugPrinted = 0
    if verbose: 
        print("Detecting peaks with PeakBot")
        print("  | .. loading PeakBot model '%s'"%(modelPath))
        print("  | .. detecting peaks in the areas in the folder '%s'"%(pathFrom))
        
    model = tf.keras.models.load_model(modelPath, custom_objects = {"batch_iou": batch_iou})
    
    for fi in tqdm.tqdm(os.listdir(pathFrom), desc="  | .. detecting chromatographic peaks", disable=not verbose):
        l = pickle.load(open(pathFrom+"/"+fi, "rb"))
        lcmsArea = l["lcmsArea"]
        for j in range(lcmsArea.shape[0]):
            m = np.max(lcmsArea[j,:,:])
            if m > 0:
                lcmsArea[j,:,:] = lcmsArea[j,:,:] / m

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

                    peaks.append([prt, pmz, prtstart, prtend, pmzstart, pmzend])
                    
                except:
                    import traceback
                    traceback.print_exc()
                    errors += 1
            elif ptyp == 4:
                backgrounds += 1
            elif ptyp == 5:
                walls += 1
            else: 
                errors += 1

    if verbose:
        print("  | .. %d local maxima analyzed"%(peaksDone))
        print("  | .. of these %7d (%5.1f%%) are chromatographic peaks"%(len(peaks) , 100 * len(peaks)  / peaksDone))
        print("  | .. of these %7d (%5.1f%%) are backgrounds          "%(backgrounds, 100 * backgrounds / peaksDone))
        print("  | .. of these %7d (%5.1f%%) are walls                "%(walls      , 100 * walls       / peaksDone))
        if errors>0:
            print("  | .. encountered %d errors"%(errors))
        print("  | .. took %.1f seconds"%(toc("running peakbot")))
        print("")
        
    return peaks

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
            rt, mz, inte, rtstart, rtend, mzstart, mzend = p[0], p[1], 0, p[2], p[3], p[4], p[5]
            fout.write('<feature id="%s">\n'%j)
            fout.write('  <position dim="0">%f</position>\n'%rt)
            fout.write('  <position dim="1">%f</position>\n'%mz)
            fout.write('  <intensity>%f</intensity>\n'%inte)
            fout.write('  <quality dim="0">0</quality>\n')
            fout.write('  <quality dim="1">0</quality>\n')
            fout.write('  <overallquality>0</overallquality>\n')
            fout.write('  <charge>1</charge>\n')
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
def exportPeakBotResultsTSV(peaks, toFile):
    with open(toFile, "w") as fout:
        fout.write("\t".join(["Num", "RT", "MZ", "RtStart", "RtEnd", "MzStart", "MzEnd"]))
        fout.write("\n")

        for j, p in enumerate(peaks):
            rt, mz, rtstart, rtend, mzstart, mzend = p[0], p[1], p[2], p[3], p[4], p[5]
            fout.write("\t".join((str(s) for s in [j, rt, mz, rtstart, rtend, mzstart, mzend])))
            fout.write("\n")

@timeit
def exportAreasAsFigures(pathFrom, toFolder, model = None, maxExport = 1E9, threshold=0.01):
    cur = 1
    import plotnine as p9
    import pandas as pd
    for fi in tqdm.tqdm(os.listdir(pathFrom)):
        l = pickle.load(open(pathFrom+"/"+fi, "rb"))
        lcmsArea = l["lcmsArea"]
        areaRTs  = l["areaRTs"]
        areaMZs  = l["areaMZs"]
        lcmsArea = np.expand_dims(lcmsArea, len(lcmsArea.shape))
        for j in range(lcmsArea.shape[0]):
            m = np.max(lcmsArea[j,:,:,:])
            if m > 0:
                lcmsArea[j,:,:,:] = lcmsArea[j,:,:,:] / m
        
        pred = None
        if model is not None:
            pred = model.predict(lcmsArea)
            
        for j in range(lcmsArea.shape[0]):
            
            maxExport = maxExport - 1
            if maxExport < 0:
                return 

            p9.options.figure_size = (14,5)

            rts = []
            mzs = []
            ints = []
            alphas = []
            typs = []
            
            for rowi in range(lcmsArea.shape[1]):
                for coli in range(lcmsArea.shape[2]):
                    if lcmsArea[j, rowi, coli, 0] > threshold:
                        rts.append(areaRTs[j, rowi])
                        mzs.append(areaMZs[j, coli])
                        ints.append(lcmsArea[j, rowi, coli, 0])
                        typs.append("raw (>%f)"%threshold)
                        alphas.append(0.25)

            dat = pd.DataFrame({"rts": rts, "mzs": mzs, "ints": ints, "typ": typs, "alpha": alphas})
            ## Plot LCMS area and mask
            plot = (p9.ggplot(dat, p9.aes("rts", "mzs", alpha="alpha", colour="ints")) + 
                    p9.geom_point() + 
                    p9.facet_wrap("~typ", ncol=6) +
                    p9.xlim(areaRTs[j,0], areaRTs[j, areaRTs.shape[1]-1]) + p9.ylim(areaMZs[j,0], areaMZs[j, areaMZs.shape[1]-1])
                   )
            ## Plot bounding-box and center
            plot = (plot + p9.scale_alpha(guide = None))
            
            rt, mz, rtLeftBorder, rtRightBorder, mzLowest, mzHighest = -1, -1, -1, -1, -1, -1
            apexRTInd, apexMZInd, rtLeftBorderInd, rtRightBorderInd, mzLowestInd, mzHighestInd = [int(i) for i in l["gdProps"][j,:]]
            if 0 <= apexRTInd <= l["areaRTs"].shape[1]:
                rt = l["areaRTs"][j,apexRTInd]
            if 0 <= rtLeftBorderInd <= l["areaRTs"].shape[1]:
                rtLeftBorder = l["areaRTs"][j,rtLeftBorderInd]
            if 0 <= rtRightBorderInd <= l["areaRTs"].shape[1]:
                rtRightBorder = l["areaRTs"][j,rtRightBorderInd]
            
            if 0 <= apexMZInd <= l["areaMZs"].shape[1]:
                mz = l["areaMZs"][j,apexMZInd]
            if 0 <= mzLowestInd <= l["areaMZs"].shape[1]:
                mzLowest = l["areaMZs"][j,mzLowestInd]
            if 0 <= mzHighestInd <= l["areaMZs"].shape[1]:
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
                                                    color="firebrick") )

                    except:
                        import traceback
                        traceback.print_exc()
                        errors += 1
                    
            plot = (plot + 
                    p9.ggtitle("RT %.2f (box %.2f - %.2f)\nMZ %.4f (box %.4f - %.4f, %.1f ppm)\nType '%s'"%(prt, prtstart, prtend, pmz, pmzstart, pmzend, (pmzend-pmzstart)*1E6/pmz, ["Single peak", "Isomers earlier and later", "Isomer earlier", "Isomer later", "Background", "Wall"][np.argmax(np.array(pred[0][j,:]))])))
            
            p9.ggsave(plot=plot, filename="%s/%d.png"%(toFolder, cur), height=7, width=12)
            cur += 1




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


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
            fout.write('    <pt x="%f" y="%f" />\n'%(peaks[j,8] , peaks[j, 3]*(1.-peaks[j,5]/2/1.E6)))
            fout.write('    <pt x="%f" y="%f" />\n'%(peaks[j,8] , peaks[j, 3]*(1.+peaks[j,5]/2/1.E6)))
            fout.write('    <pt x="%f" y="%f" />\n'%(peaks[j,11], peaks[j, 3]*(1.+peaks[j,5]/2/1.E6)))
            fout.write('    <pt x="%f" y="%f" />\n'%(peaks[j,11], peaks[j, 3]*(1.-peaks[j,5]/2/1.E6)))
            fout.write('  </convexhull>\n')
            fout.write('</feature>\n')

        fout.write('    </featureList>\n')
        fout.write('  </featureMap>\n')
@timeit
def exportLocalMaximaAsTSV(foutFile, peaks):
    with open(foutFile, "w") as fout:
        fout.write("\t".join(["Num", "RT", "MZ", "RtStart", "RtEnd", "MzStart", "MzEnd"]))
        fout.write("\n")

        for j in range(peaks.shape[0]):
            fout.write("\t".join((str(s) for s in [j, peaks[j,2], peaks[j,3], peaks[j,8], peaks[j,11], peaks[j, 3]*(1.-peaks[j,5]/2/1.E6), peaks[j, 3]*(1.+peaks[j,5]/2/1.E6)])))
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



            
            
            

            

            

