
from peakbot.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary
import peakbot

import os
import pathlib
import pickle
import tqdm
import random

import numpy as np


def shuffleResultsSampleNames(exportPath, instancePrefix = None, verbose = False):
    
    if instancePrefix is None:
        instancePrefix = peakbot.Config.INSTANCEPREFIX
    
    tic("shuffling")
    if verbose:
        print("Shuffling the test instances (batch name shuffling)")
    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    if verbose: 
        print("  | .. there are %d files"%(len(files)))
    
    unused = [t for t in range(len(files))]
    random.shuffle(unused)
    for i in range(len(files)):
        j = unused[0]
        del unused[0]
        
        os.rename(files[i], os.path.join(pathlib.Path(files[i]).parent.resolve(), "temp%d.pickle"%i))
    
    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    for i in range(len(files)):
        os.rename(files[i], files[i].replace("temp", instancePrefix))
        
    if verbose:
        print("  | .. took %.1f seconds"%toc("shuffling"))
        print("")
    

        
def shuffleResults(exportPath, steps = 1E5, samplesToExchange = 50, instancePrefix = None, verbose = False):
    
    if instancePrefix is None:
        instancePrefix = peakbot.Config.INSTANCEPREFIX
        
    tic("shuffling")
    if verbose:
        print("Shuffling the test instances (inter-batch shuffling)")
    files = [os.path.join(exportPath, f) for f in os.listdir(exportPath) if os.path.isfile(os.path.join(exportPath, f))]
    if verbose: 
        print("  | .. there are %d files"%(len(files)))
    
    with tqdm.tqdm(total = steps, desc="  | .. shuffling", disable=not verbose) as t:
        while steps > 0:
            filea = files[random.randint(0, len(files)-1)]
            fileb = files[random.randint(0, len(files)-1)]

            if filea == fileb:
                continue
            
            with open(filea, "rb") as temp:
                a = pickle.load(temp)
            with open(fileb, "rb") as temp:
                b = pickle.load(temp)

            samplesA = a["LCHRMSArea"].shape[0]
            samplesB = b["LCHRMSArea"].shape[0]
            
            cExchange = min(min(samplesA, samplesB), samplesToExchange)
            
            beginA = random.randint(0, samplesA - cExchange)
            beginB = random.randint(0, samplesB - cExchange)
            
            for k in a.keys():
                if isinstance(a[k], np.ndarray) and len(a[k].shape)==2:
                    temp = a[k][beginA:(beginA+cExchange),:]
                    a[k][beginA:(beginA+cExchange),:] = b[k][beginB:(beginB+cExchange),:]
                    b[k][beginB:(beginB+cExchange),:] = temp
                    
                elif isinstance(a[k], np.ndarray) and len(a[k].shape)==3:
                    temp = a[k][beginA:(beginA+cExchange),:,:]
                    a[k][beginA:(beginA+cExchange),:,:] = b[k][beginB:(beginB+cExchange),:,:]
                    b[k][beginB:(beginB+cExchange),:,:] = temp
                    
                elif isinstance(a[k], np.ndarray) and len(a[k].shape)==4:
                    temp = a[k][beginA:(beginA+cExchange),:,:,:]
                    a[k][beginA:(beginA+cExchange),:,:,:] = b[k][beginB:(beginB+cExchange),:,:,:]
                    b[k][beginB:(beginB+cExchange),:,:,:] = temp
                
                elif isinstance(a[k], list):
                    temp = a[k][beginA:(beginA+cExchange)]
                    a[k][beginA:(beginA+cExchange)] = b[k][beginB:(beginB+cExchange)]
                    b[k][beginB:(beginB+cExchange)] = temp
            
            assert samplesA == a["LCHRMSArea"].shape[0] and samplesB == b["LCHRMSArea"].shape[0]
            
            with open(filea, "wb") as temp:
                pickle.dump(a, temp)
            with open(fileb, "wb") as temp:
                pickle.dump(b, temp)

            steps = steps - 1
            t.update()
        
    if verbose:
        print("  | .. took %.1f seconds"%toc("shuffling"))
        print("")
    