import peakbot.core
import peakbot.cuda
from peakbot.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary, TabLog

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
                          
                          updateToLocalPeakProperties = False, 
                          RTpeakWidth = None, minApexBorderRatio = None, minIntensity = None, 
                          maxRTOffset = 5, maxMZOffset = 10,
                          
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
    
    if verbose: 
        print("Generating train examples ")
        print("  | .. batchSize: %d"%(batchSize))
        print("  | .. rtSlices %d x mzSlices %d"%(rtSlices, mzSlices))
        print("  | .. %d chromatographic peaks, %d walls, %d backgrounds"%(len(peaks), len(walls), len(backgrounds)))
        print("  | .. maximum population of %d per training example"%(maxPopulation))
        print("  | .. generating %d training examples"%(nTestExamples))
        print("  | .. exporting to '%s'"%(exportPath))
        if updateToLocalPeakProperties:
            print("  | .. Restricting to features detected in this chromatogram")
            print("  | .. .. RT peak width %.1f - %.1f"%(RTpeakWidth[0], RTpeakWidth[1]))
            print("  | .. .. minimum border to apex ratio %.1f"%(minApexBorderRatio))
            print("  | .. .. minimum apex intensity %.1g"%(minIntensity))
            print("  | .. .. maximum RT difference %.1f seconds"%(maxRTOffset))
            print("  | .. .. maximum mz difference %.1f ppm"%(maxMZOffset))
        print("  | .. device:", str(cuda.get_current_device().name))
        print("  | .. blockdim:", blockdim, "griddim:", griddim)
        print("  | .. instance prefix: '%s'"%(instancePrefix))
        print("  |")
        
    peakbot.cuda.initializeCUDAFunctions(rtSlices = rtSlices, mzSlices = mzSlices)
    cuda.synchronize()
    
    tic("sampleGeneratingTrainingDataset")

    if updateToLocalPeakProperties:
        tic("restricting")
        if verbose: 
            print("  | Restricting reference peaks to putative peaks also detected in this chromatogram")
        
        def _matchPeaks(peaksFile, peaksRef, maxRTOffset, maxMZOffset):
            lInd = cuda.grid(1)
            lLen = cuda.gridsize(1)

            for peakRefInd in range(peaksRef.shape[0]):
                bestMatchInd = -1
                bestMatchDiff = -1
                for peakFileInd in range(lInd, peaksFile.shape[0], lLen):
                    if peaksFile[peakFileInd, 7] >= 1:
                        rtdiff = abs(peaksRef[peakRefInd, 0] - peaksFile[peakFileInd, 2])
                        mzdiff = abs(peaksRef[peakRefInd, 1] - peaksFile[peakFileInd, 3])*1E6/peaksFile[peakFileInd,3]
                        if rtdiff <= maxRTOffset and mzdiff <= maxMZOffset:
                            peaksRef[peakRefInd, 5] = peakFileInd
        matchPeaks = cuda.jit()(_matchPeaks)
        
        oriRefPeaks = len(peaks)
        rPeaks = np.zeros(shape = (len(peaks), 6), dtype = np.float32)
        for pi, p in enumerate(peaks):
            rPeaks[pi, 0:5] = np.array(p, dtype=np.float32)
        d_rPeaks = cuda.to_device(rPeaks)
        fpeaks, maximaProps, maximaPropsAll = peakbot.cuda.preProcessChromatogram(
                mzxml, fileIdentifier, 
                intraScanMaxAdjacentSignalDifferencePPM = intraScanMaxAdjacentSignalDifferencePPM,
                interScanMaxSimilarSignalDifferencePPM = interScanMaxSimilarSignalDifferencePPM,
                RTpeakWidth = RTpeakWidth,
                minApexBorderRatio = minApexBorderRatio,
                minIntensity = minIntensity,
                exportLocalMaxima = None,
                exportPath = None, 
                blockdim = blockdim,
                griddim  = griddim, 
                verbose = verbose, verbosePrefix = "  | ")
        
        d_fPeaks = cuda.to_device(fpeaks)
        matchPeaks[griddim, blockdim](d_fPeaks, d_rPeaks, maxRTOffset, maxMZOffset)
        rPeaks = d_rPeaks.copy_to_host()
        d_rPeaks = None
        peaks = rPeaks[rPeaks[:,5]>0,:]
        
        for ri in range(peaks.shape[0]):
            oInd = int(peaks[ri,5])
            peaks[ri, 0:5] = fpeaks[oInd, [2,3,8,11,5]]
        peaks = np.ascontiguousarray(peaks[:,0:5])
        
        if verbose:
            print("  | .. using %d peaks of the reference (%d) also detected in this chromatogram"%(peaks.shape[0], oriRefPeaks))
            print("  | .. took %.1f seconds"%(toc("restricting")))
            print("  | ")
        
        TabLog().addData(fileIdentifier, "n local features", "%d (/%d)"%(peaks.shape[0], oriRefPeaks))
        
        
        
        tic("restricting")
        ## walls: list of arrays with the elements:
        ##        mz, rtstart, rtend, mzDeviation,    use
        def _checkWalls(mzs, ints, times, peaksCount, walls, maxmzDiffPPMAdjacentProfileSignal):
            lInd = cuda.grid(1)
            lLen = cuda.gridsize(1)
            
            for wallInd in range(lInd, walls.shape[0], lLen):
                foundMZ    = 0
                totalScans = 0
                mzSum      = 0
                devSum     = 0
                foundDev   = 0
                
                for scanInd in range(mzs.shape[0]):
                    if walls[wallInd,1] <= times[scanInd] <= walls[wallInd,2]:
                        totalScans += 1
                        peakInd = peakbot.cuda._findMostSimilarMZ_kernel(mzs, ints, times, peaksCount, scanInd, walls[wallInd,0]*(1-maxMZOffset/1E6), walls[wallInd,0]*(1+maxMZOffset/1E6), walls[wallInd,0])
                        if peakInd > -1:
                            
                            ok = False
                            while not ok:
                                if peakInd-1 >= 0 and (mzs[scanInd, peakInd]-mzs[scanInd,peakInd-1])/mzs[scanInd,peakInd]*1E6 <= maxmzDiffPPMAdjacentProfileSignal and ints[scanInd, peakInd-1] > ints[scanInd, peakInd]:
                                    peakInd -= 1
                                elif peakInd+1 < peaksCount[scanInd] and (mzs[scanInd,peakInd+1]-mzs[scanInd,peakInd])/mzs[scanInd,peakInd]*1E6 <= maxmzDiffPPMAdjacentProfileSignal and ints[scanInd, peakInd+1] > ints[scanInd, peakInd]:
                                    peakInd += 1
                                else:
                                    ok = True
                            
                            foundMZ += 1
                            mzSum += mzs[scanInd, peakInd]
                            
                            mzdev = peakbot.cuda._gradientDescendMZProfile_kernel(mzs, ints, times, peaksCount, scanInd, peakInd, maxmzDiffPPMAdjacentProfileSignal)
                            if mzdev > 0:
                                foundDev += 1
                                devSum += mzdev
                                
                if foundMZ / totalScans > 0.9:
                    walls[wallInd, 4] = 1
                    walls[wallInd, 0] = mzSum / foundMZ
                    walls[wallInd, 3] = devSum / foundDev
        checkWalls = cuda.jit()(_checkWalls)
        
        mzs, ints, times, peaksCount = mzxml.convertChromatogramToNumpyObjects(verbose = False)
        d_mzs, d_ints, d_times, d_peaksCount = peakbot.cuda.copyToDevice(mzs, ints, times, peaksCount)
        
        oriRefWalls = len(walls)
        rWalls = np.zeros(shape=(len(walls), 5), dtype = np.float32)
        for wi, w in enumerate(walls):
            rWalls[wi,0:4] = np.array(w, dtype=np.float32)
        d_rWalls = cuda.to_device(rWalls)
        checkWalls[griddim, blockdim](d_mzs, d_ints, d_times, d_peaksCount, d_rWalls, intraScanMaxAdjacentSignalDifferencePPM)
        rWalls = d_rWalls.copy_to_host()
        d_rWalls = None
        
        walls = rWalls[rWalls[:,4]>0,:]
        
        if verbose:
            print("  | .. using %d walls of the reference (%d) also detected in this chromatogram"%(walls.shape[0], oriRefWalls))
            print("  | .. took %.1f seconds"%(toc("restricting")))
            print("  | ")
        
        TabLog().addData(fileIdentifier, "n local walls", "%d (/%d)"%(walls.shape[0], oriRefWalls))
        
        
        
        tic("restricting")
        ## backgrounds: list of arrays with the elements:
        ##        rtStart, rtEnd, mzLow, mzHigh
        ## copy to 
        ## backgrounds: list of arrays with the elements:
        ##        rt, mz, mzdev
        
        oriRefBackgrounds = len(backgrounds)
        maximaPropsAll[:,7] = 0
        for bgInd, background in enumerate(backgrounds):
            background[0]=0
            crts = maximaPropsAll[:,2]
            cmzs = maximaPropsAll[:,3]
            maximaPropsAll[:,7] = np.logical_or(maximaPropsAll[:,7], np.logical_and(np.logical_and(background[0] <= crts, crts <= background[1]), np.logical_and(background[2] <= cmzs, cmzs <= background[3])))
        backgrounds = maximaPropsAll[maximaPropsAll[:,7]>0,:]
        backgrounds = backgrounds[:,[2,3,5]]
        
        
        if verbose:
            print("  | .. using %d backgrounds of the reference (%d) also detected in this chromatogram"%(backgrounds.shape[0], oriRefBackgrounds))
            print("  | .. took %.1f seconds"%(toc("restricting")))
            print("  | ")
        
        TabLog().addData(fileIdentifier, "n local walls", "%d (/%d)"%(backgrounds.shape[0], oriRefBackgrounds))
        
        
        d_mzs = None; d_ints = None; d_times = None; d_peaksCount = None
        
    
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
    ##        rtStart, rtEnd, mzLow, mzHigh
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
                mpRT       = 0
                mpMZ       = 0
                mpRTLeft   = 0
                mpRTRight  = 0 
                mpMZLow    = 0
                mpMZHigh   = 0
                typR = xoroshiro128p_uniform_float32(rng_states, lInd)*5
                typ = -1  #[isFullPeak, isPartPeak, hasCoelutingPeakLeft, hasCoelutingPeakRight, isBackground, isWall]
                
                ## main info in center of the area
                if populationInd == 0:
                    
                    ## Peak (change is higher since it is divided into four subgroups [single, left or right isomer, left and right isomer])              
                    if typR <= 4:
                        typ       = 0
                        ind       = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(peaks.shape[0]-1))
                        centerRT  = peaks[ind, 0]
                        centerMZ  = peaks[ind, 1]
                        mzDevPPM  = peaks[ind, 4] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                        devMult   = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                        mzLow     = peaks[ind, 1] * (1 - mzDevPPM * devMult / 2 / 1E6)
                        mzHigh    = peaks[ind, 1] * (1 + mzDevPPM * devMult / 2 / 1E6)
                        
                        mpRT      = centerRT
                        mpMZ      = peaks[ind,1]
                        mpRTLeft  = peaks[ind,2]
                        mpRTRight = peaks[ind,3]
                        mpMZLow   = peaks[ind, 1] * (1 - mzDevPPM / 2 / 1E6)
                        mpMZHigh  = peaks[ind, 1] * (1 + mzDevPPM / 2 / 1E6)
                    
                    ## Wall
                    elif typR <= 5:
                        typ       = 4
                        ind       = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(walls.shape[0]-1))
                        centerRT  = walls[ind, 1] + (walls[ind, 2]-walls[ind, 1]) * xoroshiro128p_uniform_float32(rng_states, lInd)
                        centerMZ  = walls[ind, 0]
                        mzDevPPM  = walls[ind, 3] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                        devMult   = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                        mzLow     = walls[ind, 0] * (1 - mzDevPPM * devMult / 2 / 1E6)
                        mzHigh    = walls[ind, 0] * (1 + mzDevPPM * devMult / 2 / 1E6)
                        
                        mpRT      = centerRT
                        mpMZ      = walls[ind,0]
                        mpRTLeft  = walls[ind,1]
                        mpRTRight = walls[ind,2]
                        mpMZLow   = walls[ind, 0] * (1 - mzDevPPM / 2 / 1E6)
                        mpMZHigh  = walls[ind, 0] * (1 + mzDevPPM / 2 / 1E6)
                        
                    ## Background
                    elif typR <= 6:
                        typ       = 5
                        ind       = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(backgrounds.shape[0]-1))
                        centerRT  = backgrounds[ind, 0]
                        centerMZ  = backgrounds[ind, 1]
                        mzDevPPM  = backgrounds[ind, 2] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                        devMult   = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                        mzLow     = backgrounds[ind, 1] * (1 - mzDevPPM * devMult / 2 / 1E6)
                        mzHigh    = backgrounds[ind, 1] * (1 + mzDevPPM * devMult / 2 / 1E6)
                        
                        mpRT      = centerRT
                        mpMZ      = peaks[ind,1]
                        mpRTLeft  = centerRT - 8 * meanDifferenceScans
                        mpRTRight = centerRT + 8 * meanDifferenceScans
                        mpMZLow   = backgrounds[ind, 1] * (1 - mzDevPPM / 2 / 1E6)
                        mpMZHigh  = backgrounds[ind, 1] * (1 + mzDevPPM / 2 / 1E6)
                
                ## distraction augmentation not in center of the area
                if populationInd > 0:
                    ## Peak
                    if typR <= 4:
                        typ = 0
                        tries = 40
                        ok = False
                        while tries > 0 and not ok:
                            tries    = tries - 1
                            sind     = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(peaks.shape[0]-1))
                            rtOffset = meanDifferenceScans * rtSlices/2 * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1)
                            centerRT = peaks[sind, 0] + rtOffset
                            mzOffset = peaks[sind, 1]/1E6 * peaks[sind, 4] * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1) * 2
                            mzDevPPM = peaks[sind, 4] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                            devMult  = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                            mzLow    = (peaks[sind, 1] - mzOffset) * (1 - mzDevPPM * devMult / 2 / 1E6)
                            mzHigh   = (peaks[sind, 1] - mzOffset) * (1 + mzDevPPM * devMult / 2 / 1E6)
                            
                            crtDiff  = mpRT - peaks[sind, 0]
                            cmzDiff  = mpMZ - peaks[sind, 1]
                            centerMZ = peaks[sind, 1] + mzOffset
                            
                            if ((mzOffset > 0 and (peaks[sind, 1] + mzOffset) * (1 - mzDevPPM / 2 / 1E6)+cmzDiff > mpMZHigh) or \
                                (mzOffset <= 0 and (peaks[sind, 1] + mzOffset) * (1 + mzDevPPM / 2 / 1E6)+cmzDiff < mpMZLow)) and \
                               ((peaks[sind, 0] < mpRT and (peaks[sind,3] < mpRTLeft or 0.15 <= (peaks[sind,3]+crtDiff-rtOffset - mpRTLeft) / (mpRT - (peaks[sind,0]+crtDiff-rtOffset)) <= 0.75)) or \
                                (peaks[sind, 0] > mpRT and (mpRTRight < peaks[sind,2] or 0.15 <= (mpRTRight - peaks[sind,2]+crtDiff-rtOffset) / ((peaks[sind,0]+crtDiff-rtOffset) - mpRT) <= 0.75))):
                                ok = True
                        
                        if not ok:
                            continue
                    
                    ## Wall
                    elif typR <= 5:
                        tries = 40
                        ok = False
                        while tries > 0 and not ok:
                            tries    = tries - 1
                            sind     = round(xoroshiro128p_uniform_float32(rng_states, lInd)*walls.shape[0])
                            centerRT = walls[sind, 1] + (walls[sind, 2]-walls[sind, 1]) * xoroshiro128p_uniform_float32(rng_states, lInd)
                            mzOffset = walls[sind, 0]/1E6 * walls[sind, 3] * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1) * 2
                            mzDevPPM = walls[sind, 3] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                            devMult  = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                            mzLow    = (walls[sind, 0] - mzOffset) * (1 - mzDevPPM * devMult / 2 / 1E6)
                            mzHigh   = (walls[sind, 0] - mzOffset) * (1 + mzDevPPM * devMult / 2 / 1E6)
                            
                            cmzDiff  = mpMZ - walls[sind, 0]
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
                            tries    = tries - 1
                            sind     = round(xoroshiro128p_uniform_float32(rng_states, lInd)*backgrounds.shape[0])
                            rtOffset = meanDifferenceScans * rtSlices/2 * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1)
                            centerRT = backgrounds[sind, 0] + rtOffset
                            mzOffset = backgrounds[sind, 1]/1E6 * backgrounds[sind, 2] * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1) * 2
                            mzDevPPM = backgrounds[sind, 2] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                            devMult  = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                            mzLow    = (backgrounds[sind, 1] - mzOffset) * (1 - mzDevPPM * devMult / 2 / 1E6)
                            mzHigh   = (backgrounds[sind, 1] - mzOffset) * (1 + mzDevPPM * devMult / 2 / 1E6)
                            
                            crtDiff  = mpRT - backgrounds[sind, 0]
                            cmzDiff  = mpMZ - backgrounds[sind, 1]
                            centerMZ = backgrounds[sind, 1] + mzOffset
                            
                            if ((mzOffset > 0 and (backgrounds[sind, 1] + mzOffset) * (1 - mzDevPPM / 2 / 1E6)+cmzDiff > mpMZHigh) or \
                                (mzOffset <= 0 and (backgrounds[sind, 1] + mzOffset) * (1 + mzDevPPM / 2 / 1E6)+cmzDiff < mpMZLow)) and \
                               ((backgrounds[sind, 0] < mpRT and backgrounds[sind,0] < mpRTLeft) or \
                                (backgrounds[sind, 0] > mpRT and mpRTRight < backgrounds[sind,0])):
                                ok = True
                        if not ok:
                            continue
                
                ## calculate supports for the addition and save them
                for y in range(mzSlices):
                    supports[y] = mzLow + (mzHigh-mzLow)*(y/(mzSlices-1))
                    areaMZs[instanceInd, y] = supports[y]

                ## combine area with previous area
                maxInt = peakbot.cuda._renovateArea_kernel(mzs, ints, times, peaksCount, centerRT, centerMZ, supports, temp, 0, areaTimes, (supports[1]-supports[0])*8)
                
                ## scale area
                ## and save area times
                r = xoroshiro128p_uniform_float32(rng_states, lInd) * maxInt + 1
                maxInt = max(maxInt, 1)
                for x in range(rtSlices):
                    if populationInd == 0:
                        areaRTs[instanceInd, x] = areaTimes[x]
                    for y in range(mzSlices):
                        if temp[0,x,y] > 0:
                            instances[instanceInd, x, y] += temp[0,x,y] * xoroshiro128p_uniform_float32(rng_states, lInd)*randomnessFactor * r / maxInt
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
                            devMult   = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                            mzLow = peaks[sind, 1] * (1 - mzDevPPM * devMult / 2 / 1E6)
                            mzHigh = peaks[sind, 1] * (1 + mzDevPPM * devMult / 2 / 1E6)
                            
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
                            maxInt = peakbot.cuda._renovateArea_kernel(mzs, ints, times, peaksCount, peaks[sind, 0] + rtOffset, mpMZ, dsupports, temp, 0, dareaTimes, (supports[1]-supports[0])*8)

                            ## scale area
                            ## and save area times
                            r = xoroshiro128p_uniform_float32(rng_states, lInd) * maxInt + 1
                            for x in range(rtSlices):
                                for y in range(mzSlices):
                                    if temp[0,x,y] > 0:
                                        instances[instanceInd, x, y] += temp[0,x,y] * xoroshiro128p_uniform_float32(rng_states, lInd)*randomnessFactor * r / maxInt
                            
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
                            devMult   = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                            mzLow = peaks[sind, 1] * (1 - mzDevPPM * devMult / 2 / 1E6)
                            mzHigh = peaks[sind, 1] * (1 + mzDevPPM * devMult / 2 / 1E6)
                            
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
                            maxInt = peakbot.cuda._renovateArea_kernel(mzs, ints, times, peaksCount, peaks[sind, 0] - rtOffset, mpMZ, dsupports, temp, 0, dareaTimes, (supports[1]-supports[0])*8)
                            
                            ## scale area
                            ## and save area times
                            r = xoroshiro128p_uniform_float32(rng_states, lInd) * maxInt + 1
                            for x in range(rtSlices):
                                for y in range(mzSlices):
                                    if temp[0,x,y] > 0:
                                        instances[instanceInd, x, y] += temp[0,x,y] * xoroshiro128p_uniform_float32(rng_states, lInd)*randomnessFactor * r / maxInt
                                
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
                    
                    if typ < 4:
                        centers[instanceInd, 0] = peakbot.cuda.getBestMatch_kernel(areaTimes, mpRT)
                        centers[instanceInd, 1] = peakbot.cuda.getBestMatch_kernel(supports , mpMZ)
                        boxes[instanceInd, 0]   = peakbot.cuda.getBestMatch_kernel(areaTimes, mpRTLeft)
                        boxes[instanceInd, 1]   = peakbot.cuda.getBestMatch_kernel(areaTimes, mpRTRight)
                        boxes[instanceInd, 2]   = peakbot.cuda.getBestMatch_kernel(supports , mpMZLow)
                        boxes[instanceInd, 3]   = peakbot.cuda.getBestMatch_kernel(supports , mpMZHigh)
                        
                    else:
                        centers[instanceInd, 0] = 0
                        centers[instanceInd, 1] = 0
                        boxes[instanceInd, 0]   = 0
                        boxes[instanceInd, 1]   = 0
                        boxes[instanceInd, 2]   = 0
                        boxes[instanceInd, 3]   = 0
        
            ## Scale to a maximum intensity of 1
            mVal = 0
            for x in range(rtSlices):
                for y in range(mzSlices):
                    mVal = max(mVal, instances[instanceInd,x,y])
            if mVal > 0:
                for x in range(rtSlices):
                     for y in range(mzSlices):
                        instances[instanceInd,x,y] /= mVal
    generateTestExamples = cuda.jit()(_generateTestExamples)
    
    
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
    d_peaks       = cuda.to_device(peaks)
    d_walls       = cuda.to_device(walls)
    d_backgrounds = cuda.to_device(backgrounds)
    
    genSamples = blockdim[0] * blockdim[1] * griddim[0] * griddim[1]
    
    ## local instances
    instances   = np.zeros((genSamples, rtSlices, mzSlices), np.float32)
    single      = np.zeros((genSamples, rtSlices, mzSlices), np.float32)
    areaRTs     = np.zeros((genSamples, rtSlices), np.float32)
    areaMZs     = np.zeros((genSamples, mzSlices), np.float32)
    peakTypes   = np.zeros((genSamples, peakbot.Config.NUMCLASSES), np.float32)
    centers     = np.zeros((genSamples, 2), np.float32)
    boxes       = np.zeros((genSamples, 4), np.float32)
    properties  = np.zeros((genSamples, maxPopulation, 10), np.float32)
    description = ["" for i in range(genSamples)]
    
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
        print("  | .. with the current values of blockdim (%d,%d) and griddim (%d,%d) %d samples will be generated on the GPU"%(blockdim[0], blockdim[1], griddim[0], griddim[1], genSamples))
        print("  | .. size of the necessary objects is %5.1f Mb (and some more local memory is required by the kernels)"%((instances.nbytes+single.nbytes+areaRTs.nbytes+areaMZs.nbytes+peakTypes.nbytes+centers.nbytes+boxes.nbytes+properties.nbytes)/1E6))
        print("  | .. some more instances than specified by the parameters will be generated. Rounding (ceiling) causes this.")
        print("  | ")
    outVal = 0
    skipped = 0
    while os.path.exists("%s/%s%d.pickle"%(exportPath, instancePrefix, outVal)):
        outVal = outVal + 1
        skipped = skipped + 1
    if skipped > 0 and verbose:
        print("\r  | .. %6d instance batches already existed. The new ones are appended starting with id %s                           "%(skipped, skipped))

    genClasses = [0,0,0,0,0,0]
    with tqdm.tqdm(total = nTestExamples, desc="  | .. current", unit="instances", disable=not verbose) as pbar:
        ## generate ahead to support overlapping kernel execution and pickle file dumping
        generateTestExamples[griddim, blockdim](rng_states, d_mzs, d_ints, d_times, d_peaksCount, d_peaks, d_walls, d_backgrounds, d_instances, d_single, d_properties, d_areaRTs, d_areaMZs, d_peakTypes, d_centers, d_boxes, maxPopulation, intensityScales, randomnessFactor)
        genTestExamples = 0
        
        while genTestExamples < nTestExamples:
            
            cuda.synchronize()
            instances  = d_instances.copy_to_host() ; single  = d_single.copy_to_host() ; 
            areaRTs    = d_areaRTs.copy_to_host()   ; areaMZs = d_areaMZs.copy_to_host(); 
            peakTypes  = d_peakTypes.copy_to_host() ; centers = d_centers.copy_to_host(); boxes = d_boxes.copy_to_host(); 
            properties = d_properties.copy_to_host();
            
            resetVars[griddim, blockdim](d_instances, d_single, d_areaRTs, d_areaMZs, d_peakTypes, d_centers, d_boxes, d_properties)
            description = ["" for i in range(batchSize)]
            generateTestExamples[griddim, blockdim](rng_states, d_mzs, d_ints, d_times, d_peaksCount, d_peaks, d_walls, d_backgrounds, d_instances, d_single, d_properties, d_areaRTs, d_areaMZs, d_peakTypes, d_centers, d_boxes, maxPopulation, intensityScales, randomnessFactor)
            
            genClasses = [genClasses[0] + sum(peakTypes[:,0]==1), genClasses[1] + sum(peakTypes[:,1]==1), genClasses[2] + sum(peakTypes[:,2]==1),
                          genClasses[3] + sum(peakTypes[:,3]==1), genClasses[4] + sum(peakTypes[:,4]==1), genClasses[5] + sum(peakTypes[:,5]==1)]
            
            cStart = 0
            while cStart < instances.shape[0] and cStart+batchSize < instances.shape[0]:
                toDump = {"LCHRMSArea"      : instances[cStart:(cStart+batchSize)], 
                          "areaRTs"         : areaRTs[cStart:(cStart+batchSize)], 
                          "areaMZs"         : areaMZs[cStart:(cStart+batchSize)], 
                          "peakType"        : peakTypes[cStart:(cStart+batchSize)], 
                          "center"          : centers[cStart:(cStart+batchSize)], 
                          "box"             : boxes[cStart:(cStart+batchSize)], 
                          "chromatogramFile": fileIdentifier[cStart:(cStart+batchSize)]}
                pickle.dump(toDump, open("%s/%s%d.pickle"%(exportPath, instancePrefix, outVal), "wb"))
                cStart += batchSize
                outVal += 1
                genTestExamples += batchSize
                pbar.update(batchSize)
            
            
    cuda.synchronize()

        
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
    TabLog().addData(fileIdentifier, "single peaks", "%d (%.1f%%)"%(genClasses[0], 100*genClasses[0]/sum(genClasses)))
    TabLog().addData(fileIdentifier, "LeRi isomer" , "%d (%.1f%%)"%(genClasses[1], 100*genClasses[1]/sum(genClasses)))
    TabLog().addData(fileIdentifier, "Le isomer"   , "%d (%.1f%%)"%(genClasses[2], 100*genClasses[2]/sum(genClasses)))
    TabLog().addData(fileIdentifier, "Ri isomer"   , "%d (%.1f%%)"%(genClasses[3], 100*genClasses[3]/sum(genClasses)))
    TabLog().addData(fileIdentifier, "walls"       , "%d (%.1f%%)"%(genClasses[4], 100*genClasses[4]/sum(genClasses)))
    TabLog().addData(fileIdentifier, "backgrounds" , "%d (%.1f%%)"%(genClasses[5], 100*genClasses[5]/sum(genClasses)))
    TabLog().addData(fileIdentifier, "Training examples (sec)", toc("sampleGeneratingTrainingDataset"))
        
        