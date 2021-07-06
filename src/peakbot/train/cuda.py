import peakbot.core
import peakbot.cuda
from peakbot.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary

import math
import pickle
import tqdm
import os

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numba

                

def generateTestInstances(mzxml, fileIdentifier, peaks, walls, backgrounds, nTestExamples, exportPath, 
                          intraScanMaxAdjacentSignalDifferencePPM, interScanMaxSimilarSignalDifferencePPM, 
                          maxPopulation = 5, intensityScales = 20, randomnessFactor = 0.1, 
                          batchSize = None, rtSlices = None, mzSlices = None, instancePrefix = None,
                          blockdim = (32, 8), griddim = (64, 4),
                          verbose = True):
    if batchSize is None:
        batchSize = peakbot.Config.BATCHSIZE
    if rtSlices is None:
        rtSlices = peakbot.Config.RTSLICES
    if mzSlices is None:
        mzSlices = peakbot.Config.MZSLICES
    if instancePrefix is None:
        instancePrefix = peakbot.Config.INSTANCEPREFIX
    
    peakbot.cuda.initializeCUDAFunctions(rtSlices = rtSlices, mzSlices = mzSlices)
    cuda.synchronize()
    
    tic("sampleGeneratingTrainingDataset")
    
    def _resetVars(instances, single, areaRTs, areaMZs, peakTypes, centers, boxes, properties):
        
        lInd = cuda.grid(1)
        lLen = cuda.gridsize(1)
        
        for x in range(lInd, instances.shape[0], lLen):
            for y in range(instances.shape[1]):
                for z in range(instances.shape[2]):
                    instances[x,y,z] = 0
                    single[x,y,z] = 0
                areaRTs[x,y] = 0
            for y in range(areaMZs.shape[1]):
                areaMZs[x,y] = 0
            for y in range(peakTypes.shape[1]):
                peakTypes[x,y] = 0
            for y in range(centers.shape[1]):
                centers[x,y] = 0
            for y in range(boxes.shape[1]):
                boxes[x,y] = 0
            for y in range(properties.shape[1]):
                for z in range(properties.shape[2]):
                    properties[x,y,z] = 0
    resetVars = cuda.jit()(_resetVars)
    
    ## peaks: list of arrays with the elements:
    ##        rt, mz,leftRTBorder, rightRTBorder, mzDeviation
    ##
    ## walls: list of arrays with the elements:
    ##        mz, rtstart, rtend, mzDeviation
    ##
    ## backgrounds: list of arrays with the elements:
    ##        rt, mz, ppmDev
    ##
    ## Note: walls and backgrounds can be very large, rt center will be picked randomly
    ## Note: actual mz deviation of peaks will be randomized but is at least mzLowerBorder and mzUpperBorder
    def _generateTestExamples(rng_states, mzs, ints, times, peaksCount, peaks, walls, backgrounds, instances, single, properties, areaRTs, areaMZs, peakTypes, centers, boxes, maxPopulation, maxInt, randomnessFactor):
        supports = cuda.local.array(shape = mzSlices, dtype=numba.float32)
        dsupports = cuda.local.array(shape = mzSlices, dtype=numba.float32)
        areaTimes = cuda.local.array(shape = rtSlices, dtype=numba.float32)
        dareaTimes = cuda.local.array(shape = rtSlices, dtype=numba.float32)
        temp = cuda.local.array(shape = (1, rtSlices, mzSlices), dtype=numba.float32)
        
        lInd = cuda.grid(1)
        lLen = cuda.gridsize(1)
        
        s = 0
        for i in range(times.shape[0]-1):
            s = s + times[i+1]-times[i]
        meanDifferenceScans = s / times.shape[0]
                
        for instanceInd in range(lInd, instances.shape[0], lLen):            
            for populationInd in range(0, math.ceil(xoroshiro128p_uniform_float32(rng_states, lInd)*maxPopulation)):
                ind        = -1
                shiftRT    = 0
                centerRT   = 0
                centerMZ   = 0
                mzLow      = 0
                mzHigh     = 0
                mpCenterRT = 0
                mpMZ       = 0
                mpRTLeft   = 0
                mpRTRight  = 0 
                mpMZLow    = 0
                mpMZHigh   = 0
                typR = xoroshiro128p_uniform_float32(rng_states, lInd)*6
                typ = 4  #[isFullPeak, isPartPeak, hasCoelutingPeakLeft, hasCoelutingPeakRight, isBackground, isWall]
                
                
                ## main info in center of the area
                if populationInd == 0:
                    
                    ## Peak (change is higher since it is divided into four subgroups [single, left or right isomer, left and right isomer])              
                    if typR <= 4:
                        typ = 0
                        ind = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(peaks.shape[0]-1))
                        centerRT = peaks[ind, 0]
                        centerMZ = peaks[ind, 1]
                        mzDevPPM = peaks[ind, 4] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                        mzLow = peaks[ind, 1] * (1 - mzDevPPM * 3 / 2 / 1E6)
                        mzHigh = peaks[ind, 1] * (1 + mzDevPPM * 3 / 2 / 1E6)
                        
                        mpCenterRT = centerRT
                        mpMZ = peaks[ind,1]
                        mpRTLeft = peaks[ind,2]
                        mpRTRight = peaks[ind,3]
                        mpMZLow = mzLow
                        mpMZHigh = mzHigh
                    
                    ## Wall
                    elif typR <= 5:
                        typ = 4
                        ind = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(walls.shape[0]-1))
                        centerRT = walls[ind, 1] + (walls[ind, 2]-walls[ind, 1]) * xoroshiro128p_uniform_float32(rng_states, lInd)
                        centerMZ = walls[ind, 0]
                        mzDevPPM = walls[ind, 3] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                        mzLow = walls[ind, 0] * (1 - mzDevPPM * 3 / 2 / 1E6)
                        mzHigh = walls[ind, 0] * (1 + mzDevPPM * 3 / 2 / 1E6)
                        
                        mpCenterRT = centerRT
                        mpMZ = walls[ind,0]
                        mpRTLeft = walls[ind,1]
                        mpRTRight = walls[ind,2]
                        mpMZLow = mzLow
                        mpMZHigh = mzHigh
                        
                    ## Background
                    elif typR <= 6:
                        typ = 5
                        ind = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(backgrounds.shape[0]-1))
                        centerRT = backgrounds[ind, 0]
                        centerMZ = backgrounds[ind, 2]
                        mzDevPPM = backgrounds[ind, 2] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                        mzLow = backgrounds[ind, 1] * (1 - mzDevPPM * 3 / 2 / 1E6)
                        mzHigh = backgrounds[ind, 1] * (1 + mzDevPPM * 3 / 2 / 1E6)
                        
                        mpCenterRT = centerRT
                        mpMZ = peaks[ind,1]
                        mpRTLeft = centerRT - 8 * meanDifferenceScans
                        mpRTRight = centerRT + 8 * meanDifferenceScans
                        mpMZLow = mzLow
                        mpMZHigh = mzHigh
                
                ## distraction augmentation not in center of the area
                if populationInd > 0:
                    ## Peak
                    if typR <= 4:
                        typ = 0
                        tries = 40
                        ok = False
                        while tries > 0 and not ok:
                            tries = tries - 1
                            sind = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(peaks.shape[0]-1))
                            rtOffset = meanDifferenceScans * rtSlices/2 * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1)
                            centerRT = peaks[sind, 0] + rtOffset
                            mzOffset = peaks[sind, 1]/1E6 * peaks[sind, 4] * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1) * 2
                            mzDevPPM = peaks[sind, 4] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                            mzLow = (peaks[sind, 1] - mzOffset) * (1 - mzDevPPM * 3 / 2 / 1E6)
                            mzHigh = (peaks[sind, 1] - mzOffset) * (1 + mzDevPPM * 3 / 2 / 1E6)
                            
                            crtDiff = mpCenterRT - peaks[sind, 0]
                            cmzDiff = mpMZ       - peaks[sind, 1]
                            centerMZ = peaks[sind, 1] + mzOffset
                            
                            if ((mzOffset > 0 and (peaks[sind, 1] + mzOffset) * (1 - mzDevPPM / 2 / 1E6)+cmzDiff > mpMZHigh) or \
                                (mzOffset <= 0 and (peaks[sind, 1] + mzOffset) * (1 + mzDevPPM / 2 / 1E6)+cmzDiff < mpMZLow)) and \
                               ((peaks[sind, 0] < mpCenterRT and (peaks[sind,3] < mpRTLeft or 0.15 <= (peaks[sind,3]+crtDiff-rtOffset - mpRTLeft) / (mpCenterRT - (peaks[sind,0]+crtDiff-rtOffset)) <= 0.75)) or \
                                (peaks[sind, 0] > mpCenterRT and (mpRTRight < peaks[sind,2] or 0.15 <= (mpRTRight - peaks[sind,2]+crtDiff-rtOffset) / ((peaks[sind,0]+crtDiff-rtOffset) - mpCenterRT) <= 0.75))):
                                ok = True
                        
                        if not ok:
                            continue
                    
                    ## Wall
                    elif typR <= 5:
                        tries = 40
                        ok = False
                        while tries > 0 and not ok:
                            tries = tries - 1
                            sind = round(xoroshiro128p_uniform_float32(rng_states, lInd)*walls.shape[0])
                            centerRT = walls[sind, 1] + (walls[sind, 2]-walls[sind, 1]) * xoroshiro128p_uniform_float32(rng_states, lInd)
                            mzOffset = walls[sind, 0]/1E6 * walls[sind, 3] * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1) * 2
                            mzDevPPM = walls[sind, 3] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                            mzLow = (walls[sind, 0] - mzOffset) * (1 - mzDevPPM * 3 / 2 / 1E6)
                            mzHigh = (walls[sind, 0] - mzOffset) * (1 + mzDevPPM * 3 / 2 / 1E6)
                            
                            cmzDiff = mpMZ - walls[sind, 0]
                            centerMZ = walls[sind, 0] + mzOffset
                            
                            if (mzOffset > 0 and (walls[sind, 0] + mzOffset) * (1 - mzDevPPM / 2 / 1E6)+cmzDiff > mpMZHigh) or \
                               (mzOffset <= 0 and (walls[sind, 0] + mzOffset) * (1 + mzDevPPM / 2 / 1E6)+cmzDiff < mpMZLow):
                                ok = True
                        if not ok:
                            continue                            
                        
                    ## Background
                    elif typR <= 6:
                        tries = 40
                        ok = False
                        while tries > 0 and not ok:
                            tries = tries - 1
                            sind = round(xoroshiro128p_uniform_float32(rng_states, lInd)*backgrounds.shape[0])
                            rtOffset = meanDifferenceScans * rtSlices/2 * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1)
                            centerRT = backgrounds[sind, 0] + rtOffset
                            mzOffset = backgrounds[sind, 1]/1E6 * backgrounds[sind, 2] * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1) * 2
                            mzDevPPM = backgrounds[sind, 2] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                            mzLow = (backgrounds[sind, 0] - mzOffset) * (1 - mzDevPPM * 3 / 2 / 1E6)
                            mzHigh = (backgrounds[sind, 0] - mzOffset) * (1 + mzDevPPM * 3 / 2 / 1E6)
                            
                            crtDiff = mpCenterRT - backgrounds[sind, 0]
                            cmzDiff = mpMZ - backgrounds[sind, 1]
                            centerMZ = backgrounds[sind, 1] + mzOffset
                            
                            if ((mzOffset > 0 and (backgrounds[sind, 1] + mzOffset) * (1 - mzDevPPM / 2 / 1E6)+cmzDiff > mpMZHigh) or \
                                (mzOffset <= 0 and (backgrounds[sind, 1] + mzOffset) * (1 + mzDevPPM / 2 / 1E6)+cmzDiff < mpMZLow)) and \
                               ((backgrounds[sind, 0] < mpCenterRT and backgrounds[sind,0] < mpRTLeft) or \
                                (backgrounds[sind, 0] > mpCenterRT and mpRTRight < backgrounds[sind,0])):
                                ok = True
                        if not ok:
                            continue
                
                
                ## calculate supports for the addition and save them
                for y in range(mzSlices):
                    supports[y] = 0
                    supports[y] = mzLow + (mzHigh-mzLow)*(y/(mzSlices-1))
                    areaMZs[instanceInd, y] = supports[y]

                ## combine area with previous area
                maxIntVal = peakbot.cuda._renovateArea_kernel(mzs, ints, times, peaksCount, centerRT, centerMZ, supports, temp, 0, areaTimes, (supports[1]-supports[0])*8, 0, 1)
                
                ## scale area
                ## and save area times
                r = xoroshiro128p_uniform_float32(rng_states, lInd) * maxInt + 1
                for x in range(rtSlices):
                    if populationInd == 0:
                        areaRTs[instanceInd, x] = areaTimes[x]
                    for y in range(mzSlices):
                        if temp[0,x,y] > 0:
                            instances[instanceInd, x, y] += temp[0,x,y] * xoroshiro128p_uniform_float32(rng_states, lInd)*randomnessFactor * r / maxIntVal
                            if populationInd == 0:
                                single[instanceInd,x,y] = temp[0,x,y]
                
                
                ## Add isomeric compounds if the center peak is a chromatographic peak
                if populationInd == 0 and typR <= 4:
                    
                    ## add isomeric peak on the left side
                    if 1 < typR <= 2 or 2 < typR <= 3:
                        tries = 40
                        ok = False
                        while tries > 0 and not ok:
                            tries = tries - 1
                            sind = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(peaks.shape[0]-1))
                            rtOffset = meanDifferenceScans * (0.05 + xoroshiro128p_uniform_float32(rng_states, lInd))*rtSlices/3
                            mzDevPPM = peaks[sind, 4] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                            mzLow = peaks[sind, 1] * (1 - mzDevPPM * 3 / 2 / 1E6)
                            mzHigh = peaks[sind, 1] * (1 + mzDevPPM * 3 / 2 / 1E6)
                            
                            crtDiff = peaks[ind,0] - peaks[sind,0]
                            
                            if peaks[sind,0]+crtDiff-rtOffset < peaks[ind,2] and \
                                peaks[sind,3]+crtDiff-rtOffset >= peaks[ind,2] and \
                                0.15 <= (peaks[sind,3]+crtDiff-rtOffset - peaks[ind,2]) / (peaks[ind,0] - (peaks[sind,0]+crtDiff-rtOffset)) <= 0.75 :
                                ok = True
                        
                        if ok:
                            ## calculate supports for the addition
                            for t in range(mzSlices):
                                dsupports[t] = 0
                                dsupports[t] = mzLow + (mzHigh-mzLow)*(t/(mzSlices-1))
                            peakbot.cuda._renovateArea_kernel(mzs, ints, times, peaksCount, peaks[sind, 0] + rtOffset, mpMZ, dsupports, instances, instanceInd, dareaTimes, (supports[1]-supports[0])*8, 1, 1)
                            
                            total = -1
                            newInd = -1
                            for x in range(rtSlices):
                                if areaTimes[x] > -1 and peaks[ind, 2] <= areaTimes[x] <= peaks[sind,3]+crtDiff-rtOffset:
                                    s = 0
                                    for y in range(mzSlices):
                                        cmz = supports[y]
                                        if mpMZLow <= cmz <= mpMZHigh:
                                            s = s + instances[instanceInd,x,y]
                                    if total == -1 or s < total:
                                        total = s
                                        newInd = x
                            
                            if newInd >= 0:
                                boxes[instanceInd,0] = areaTimes[newInd]        ## rt left border
                            
                            ## make if a leftPeak
                            typ = 2
                    
                    ## add isomeric peak on the right side
                    if 1 < typR <= 2 or 3 < typR <= 4:
                        tries = 40
                        ok = False
                        while tries > 0 and not ok:
                            tries = tries - 1
                            sind = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(peaks.shape[0]-1))
                            rtOffset = meanDifferenceScans * (0.05 + xoroshiro128p_uniform_float32(rng_states, lInd))*rtSlices/3
                            mzDevPPM = peaks[sind, 4] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                            mzLow = peaks[sind, 1] * (1 - mzDevPPM * 3 / 2 / 1E6)
                            mzHigh = peaks[sind, 1] * (1 + mzDevPPM * 3 / 2 / 1E6)
                            
                            crtDiff = peaks[ind,0] - peaks[sind,0]
                            
                            if peaks[sind,0]+crtDiff+rtOffset > peaks[ind,3] and \
                                peaks[sind,2]+crtDiff+rtOffset <= peaks[ind,3] and \
                                0.15 <= (peaks[ind,3] - (peaks[sind,2]+crtDiff+rtOffset)) / (peaks[sind,0]+crtDiff+rtOffset - peaks[ind,0]) <= 0.75:
                                ok = True
                        
                        if ok:
                            ## calculate supports for the addition
                            for t in range(mzSlices):
                                dsupports[t] = 0
                                dsupports[t] = mzLow + (mzHigh-mzLow)*(t/(mzSlices-1))
                            peakbot.cuda._renovateArea_kernel(mzs, ints, times, peaksCount, peaks[sind, 0] - rtOffset, mpMZ, dsupports, instances, instanceInd, dareaTimes, (supports[1]-supports[0])*8, 1, 1)
                            
                            total = -1
                            newInd = -1
                            for x in range(rtSlices):
                                if areaTimes[x] > -1 and peaks[sind,2]+crtDiff+rtOffset <= areaTimes[x] <= peaks[ind, 3]:
                                    s = 0
                                    for y in range(mzSlices):
                                        cmz = supports[y]
                                        if mpMZLow <= cmz <= mpMZHigh:
                                            s = s + instances[instanceInd,x,y]
                                    if total == -1 or s < total:
                                        total = s
                                        newInd = x
                            
                            if newInd >= 0:
                                boxes[instanceInd,1] = areaTimes[newInd]        ## rt left border
                            
                            ## if there is a left isomer make it a partPeak
                            if typ == 0:
                                typ = 3
                            ## if there is no left isomer make it a rightPeak
                            elif typ == 2:
                                typ = 1
                
                ## Update center parameters
                if populationInd == 0:
                    peakTypes[instanceInd, typ] = 1
                    #updateProperties_kernel(areaRTs, areaMZs, instances, instanceInd, )
                    centers[instanceInd, 0] = peakbot.cuda.getBestMatch_kernel(areaTimes, centerRT)
                    centers[instanceInd, 1] = peakbot.cuda.getBestMatch_kernel(supports, centerMZ)
                    
                    boxes[instanceInd, 0] = peakbot.cuda.getBestMatch_kernel(areaTimes, mpRTLeft)
                    boxes[instanceInd, 1] = peakbot.cuda.getBestMatch_kernel(areaTimes, mpRTRight)
                    boxes[instanceInd, 2] = peakbot.cuda.getBestMatch_kernel(supports, mpMZLow)
                    boxes[instanceInd, 3] = peakbot.cuda.getBestMatch_kernel(supports, mpMZHigh)            
    generateTestExamples = cuda.jit()(_generateTestExamples)
    
    def _updateProperties(areaRTs, areaMZs, instances, instanceInd, rtapexInd, mzapexInd):
        
        isApex = False
        while not isApex:
            cint = instances[instanceInd, rtapexInd, mzapexInd]
            
            if rtapexInd-1 >= 0 and instances[instanceInd, rtapexInd-1, mzapexInd] > cint:
                rtapexInd = rtapexInd - 1
                
            elif rtapexInd+1 < areaRTs.shape[1] and instances[instanceInd, rtapexInd+1, mzapexInd] > cint:
                rtapexInd = rtapexInd + 1
                
            elif mzapexInd-1 >= 0 and instances[instanceInd,rtapexInd, mzapexInd-1] > cint:
                mzapexInd = mzapexInd - 1
                
            elif mzapexInd+1 < areaMZs.shape[1] and instances[instanceInd, rtapexInd, mzapexInd+1] > cint:
                mzapexInd = mzapexInd + 1
                
            elif rtapexInd-1 >= 0 and mzapexInd-1 >= 0 and instances[instanceInd, rtapexInd-1, mzapexInd-1] > cint:
                rtapexInd = rtapexInd - 1
                mzapexInd = mzapexInd - 1
                
            elif  rtapexInd+1 < areaRTs.shape[1] and mzapexInd+1 < areaMZs.shape[1] and instances[instanceInd, rtapexInd+1, mzapexInd+1] > cint:
                rtapexInd = rtapexInd + 1
                mzapexInd = mzapexInd + 1
                
            elif rtapexInd-1 >= 0 and mzapexInd+1 < areaMZs.shape[1] and instances[instanceInd, rtapexInd-1, mzapexInd+1] > cint:
                rtapexInd = rtapexInd - 1
                mzapexInd = mzapexInd + 1
                
            elif rtapexInd+1 < areaRTs.shape[1] and mzapexInd-1 >= 0 and instances[instanceInd, rtapexInd+1, mzapexInd-1] > cint:
                rtapexInd = rtapexInd + 1
                mzapexInd = mzapexInd -1
                
            else:
                isApex = True
        return rtapexInd, mzapexInd
        
    updateProperties_kernel = cuda.jit(device=True)(_updateProperties)
        
    
    
    
    if verbose: 
        print("Generating train examples ")
        print("  | .. batchSize: %d"%(batchSize))
        print("  | .. rtSlices %d x mzSlices %d"%(rtSlices, mzSlices))
        print("  | .. %d chromatographic peaks, %d walls, %d backgrounds"%(len(peaks), len(walls), len(backgrounds)))
        print("  | .. maximum population of %d per training example"%(maxPopulation))
        print("  | .. generating %d training examples"%(nTestExamples))
        print("  | .. exporting to '%s'"%(exportPath))
        print("  | .. device:", str(cuda.get_current_device().name))
        print("  | .. blockdim:", blockdim, "griddim:", griddim)
        print("  | .. instance prefix: '%s'"%(instancePrefix))
        print("  |")
    
    rng_states = create_xoroshiro128p_states(griddim[0] * griddim[1] * blockdim[0] * blockdim[1], seed=1)
    cuda.synchronize()
    
    
    if verbose:
        print("  | Converted to numpy objects")
    tic()
    mzs, ints, times, peaksCount = mzxml.convertChromatogramToNumpyObjects(verbose = verbose)
    if verbose:
        print("  | .. size of objects is %5.1f Mb"%(mzs.nbytes/1E6 + ints.nbytes/1E6 + times.nbytes/1E6 + peaksCount.nbytes/1E6))
        print("  | ")
    
    d_mzs, d_ints, d_times, d_peaksCount = peakbot.cuda.copyToDevice(mzs, ints, times, peaksCount)
    d_peaks = cuda.to_device(peaks)
    d_walls = cuda.to_device(walls)
    d_backgrounds = cuda.to_device(backgrounds)
    
    ## local instances
    instances   = np.zeros((batchSize, rtSlices, mzSlices), np.float32)
    single      = np.zeros((batchSize, rtSlices, mzSlices), np.float32)
    areaRTs     = np.zeros((batchSize, rtSlices), np.float32)
    areaMZs     = np.zeros((batchSize, mzSlices), np.float32)
    peakTypes   = np.zeros((batchSize, peakbot.Config.NUMCLASSES), np.float32)
    centers     = np.zeros((batchSize, 2), np.float32)
    boxes       = np.zeros((batchSize, 4), np.float32)
    properties  = np.zeros((batchSize, maxPopulation, 10), np.float32)
    description = ["" for i in range(batchSize)]
    
    ## gpu memory instances
    d_instances      = cuda.to_device(instances)
    d_single         = cuda.to_device(single)
    d_areaRTs        = cuda.to_device(areaRTs)
    d_areaMZs        = cuda.to_device(areaMZs)
    d_peakTypes      = cuda.to_device(peakTypes)
    d_centers        = cuda.to_device(centers)
    d_boxes          = cuda.to_device(boxes)
    d_properties     = cuda.to_device(properties)
    
    if verbose: 
        print("  | Generating %d examples in %d batches on the GPU"%(nTestExamples, math.ceil(nTestExamples/batchSize)))
    outVal = 0
    genClasses = [0,0,0,0,0,0]
    with tqdm.tqdm(total = nTestExamples, desc="  | .. current", unit="instances", disable=not verbose) as pbar: 
        for i in range(0, nTestExamples, batchSize):
            
            generateTestExamples[griddim, blockdim](rng_states, d_mzs, d_ints, d_times, d_peaksCount, d_peaks, d_walls, d_backgrounds, d_instances, d_single, d_properties, d_areaRTs, d_areaMZs, d_peakTypes, d_centers, d_boxes, maxPopulation, intensityScales, randomnessFactor)
            cuda.synchronize()

            instances = d_instances.copy_to_host(); single     = d_single.copy_to_host()    ; areaRTs = d_areaRTs.copy_to_host()
            areaMZs   = d_areaMZs.copy_to_host()  ; peakTypes  = d_peakTypes.copy_to_host() ; centers = d_centers.copy_to_host()
            boxes     = d_boxes.copy_to_host()    ; properties = d_properties.copy_to_host();
            
            resetVars[griddim, blockdim](d_instances, d_single, d_areaRTs, d_areaMZs, d_peakTypes, d_centers, d_boxes, d_properties)
            description = ["" for i in range(batchSize)]
            
            genClasses = [genClasses[0] + sum(peakTypes[:,0]==1), genClasses[1] + sum(peakTypes[:,1]==1), genClasses[2] + sum(peakTypes[:,2]==1),
                          genClasses[3] + sum(peakTypes[:,3]==1), genClasses[4] + sum(peakTypes[:,4]==1), genClasses[5] + sum(peakTypes[:,5]==1)]

            skipped = 0
            while os.path.exists("%s/%s%d.pickle"%(exportPath, instancePrefix, outVal)):
                outVal = outVal + 1
                skipped = skipped + 1
            if skipped > 0 and verbose:
                print("\r  | .. %6d instance batches already existed. The new ones are appended starting with id %s                           "%(skipped, skipped))

            toDump = {"LCHRMSArea"      : instances, 
                      "areaRTs"         : areaRTs, 
                      "areaMZs"         : areaMZs, 
                      "peakType"        : peakTypes, 
                      "center"          : centers, 
                      "box"             : boxes, 
                      "chromatogramFile": fileIdentifier}
            pickle.dump(toDump, open("%s/%s%d.pickle"%(exportPath, instancePrefix, outVal), "wb"))
            
            cuda.synchronize()

            outVal += 1
            pbar.update(batchSize)
        
    #peakbot.exportAreasAsFigures("%s/%d"%(exportPath, batchSize), ".", maxExport = 30)
    
    d_mzs = None; d_ints = None; d_times = None; d_peaksCount = None; d_peaks = None; 
    d_walls = None; d_backgrounds = None; d_instances = None; d_single = None; d_properties = None; 
    d_areaRTs = None; d_areaMZs = None; d_peakTypes = None; d_centers = None; d_boxes = None;
    cuda.defer_cleanup()
    
    if verbose:
        print("  | .. generated %d training instances"%sum(genClasses))
        print("  |    .. %7d (%4.1f%%) single peaks"                        %(genClasses[0], 100*genClasses[0]/sum(genClasses)))
        print("  |    .. %7d (%4.1f%%) peaks with earlier and later isomers"%(genClasses[1], 100*genClasses[1]/sum(genClasses)))
        print("  |    .. %7d (%4.1f%%) peaks with earlier isomers"          %(genClasses[2], 100*genClasses[2]/sum(genClasses)))
        print("  |    .. %7d (%4.1f%%) peaks with later isomers"            %(genClasses[3], 100*genClasses[3]/sum(genClasses)))
        print("  |    .. %7d (%4.1f%%) walls"                               %(genClasses[4], 100*genClasses[4]/sum(genClasses)))
        print("  |    .. %7d (%4.1f%%) backgrounds"                         %(genClasses[5], 100*genClasses[5]/sum(genClasses)))
        print("  | .. took %.1f seconds"%(toc("sampleGeneratingTrainingDataset")))
        print("")
        
        