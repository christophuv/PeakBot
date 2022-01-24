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
                          RTpeakWidth = None, minApexBorderRatio = 4, minIntensity = None,
                          maxRTOffset = 5, maxMZOffset = 10,
                          SavitzkyGolayWindowPlusMinus = 5, 

                          maxPopulation = 5, intensityScales = 10, randomnessFactor = 0.1,
                          overlapMinIOU = 0.1, overlapMaxIOU = 0.33, overlapRemain = 0.50,
                          instancePrefix = None,
                          blockdim = 32, griddim = 64,
                          verbose = True):
    batchSize = peakbot.Config.BATCHSIZE
    rtSlices = peakbot.Config.RTSLICES
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

    peakbot.cuda.initializeCUDAFunctions(SavitzkyGolayWindowPlusMinus = SavitzkyGolayWindowPlusMinus)
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
            peaks[ri, 0:5] = fpeaks[oInd, [2,3,8,11,5]] #RT, MZ, leftRTBorder, rightRTBorder, PPMdevMZProfile
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

                            meanmz, mzdev = peakbot.cuda._gradientDescendMZProfile_kernel(mzs, ints, times, peaksCount, scanInd, peakInd, maxmzDiffPPMAdjacentProfileSignal)
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
        for background in backgrounds:
            crts = maximaPropsAll[:,2]
            cmzs = maximaPropsAll[:,3]
            maximaPropsAll[:,7] = np.logical_or(maximaPropsAll[:,7]==0, np.logical_and(np.logical_and(background[0] <= crts, crts <= background[1]), np.logical_and(background[2] <= cmzs, cmzs <= background[3])))
        backgrounds = maximaPropsAll[maximaPropsAll[:,7]>0,:]
        backgrounds = backgrounds[:,[2,3,5]]
        if backgrounds.shape[0] > 10000:
            print("  | .. restricting %d backgrounds to a manageable size of 1000"%(backgrounds.shape[0]))
            backgrounds = backgrounds[np.random.choice(backgrounds.shape[0], 1000, replace=False),:]


        if verbose:
            print("  | .. using %d backgrounds of the reference (%d) also detected in this chromatogram"%(backgrounds.shape[0], oriRefBackgrounds))
            print("  | .. took %.1f seconds"%(toc("restricting")))
            print("  | ")

        TabLog().addData(fileIdentifier, "n local backgrounds", "%d (/%d)"%(backgrounds.shape[0], oriRefBackgrounds))


        d_mzs = None; d_ints = None; d_times = None; d_peaksCount = None
        d_rPeaks = None; d_fPeaks = None; d_rWalls = None;
        cuda.defer_cleanup()

    def _resetVars(instances, areaRTs, areaMZs, peakTypes, centers, boxes):

        lInd = cuda.grid(1)
        lLen = cuda.gridsize(1)

        for x in range(lInd, instances.shape[0], lLen):
            for y in range(instances.shape[1]):
                for z in range(instances.shape[2]):
                    instances[x,y,z] = 0
                areaRTs[x,y] = 0
            for y in range(areaMZs.shape[1]):
                areaMZs[x,y] = 0
            for y in range(peakTypes.shape[1]):
                peakTypes[x,y] = 0
            for y in range(centers.shape[1]):
                centers[x,y] = 0
            for y in range(boxes.shape[1]):
                boxes[x,y] = 0
    resetVars = cuda.jit()(_resetVars)

    def _iou(boxAxs, boxAxe, boxAys, boxAye, boxBxs, boxBxe, boxBys, boxBye):
        intxs = max(boxAxs, boxBxs)
        intxe = min(boxAxe, boxBxe)
        intys = max(boxAys, boxBys)
        intye = min(boxAye, boxBye)

        if intxs > intxe or intys > intye:
            return 0

        inter = (intxe - intxs) * (intye - intys)

        areaA = (boxAxe - boxAxs) * (boxAye - boxAys)
        areaB = (boxBxe - boxBxs) * (boxBye - boxBys)

        iou = inter / (areaA + areaB - inter)

        return iou
    _iou_kernel = cuda.jit(device=True)(_iou)

    def _leftNot(boxAxs, boxAxe, boxAys, boxAye, boxBxs, boxBxe, boxBys, boxBye):
        intxs = max(boxAxs, boxBxs)
        intxe = min(boxAxe, boxBxe)
        intys = max(boxAys, boxBys)
        intye = min(boxAye, boxBye)

        inter = (intxe - intxs) * (intye - intys)

        areaA = (boxAxe - boxAxs) * (boxAye - boxAys)

        leftNot = 1 - inter / areaA

        return leftNot
    _leftNot_kernel = cuda.jit(device=True)(_leftNot)

    def _rightNot(boxAxs, boxAxe, boxAys, boxAye, boxBxs, boxBxe, boxBys, boxBye):
        intxs = max(boxAxs, boxBxs)
        intxe = min(boxAxe, boxBxe)
        intys = max(boxAys, boxBys)
        intye = min(boxAye, boxBye)

        inter = (intxe - intxs) * (intye - intys)

        areaB = (boxBxe - boxBxs) * (boxBye - boxBys)

        rightNot = 1 - inter / areaB

        return rightNot
    _rightNot_kernel = cuda.jit(device=True)(_rightNot)

    def _getAsymmetricEllipse(mat, instanceInd, xcInd, ycInd, leftWidth, rightWidth, heightUp, heightDown):
        a = math.sqrt(math.pow(leftWidth,2) + math.pow(heightUp,2))
        for x in range(max(0, xcInd - leftWidth - 2), min(mat.shape[1], xcInd + 2)):
            for y in range(max(0, ycInd - 2), min(mat.shape[2], ycInd + heightUp + 2)):
                xa = x - xcInd
                ya = y - ycInd
                if math.pow(xa, 2)/math.pow(a,2) + math.pow(ya,2)/math.pow(heightUp,2) <= 0.25:
                    mat[instanceInd, x, y] = 1

        a = math.sqrt(math.pow(leftWidth,2) + math.pow(heightDown,2))
        for x in range(max(0, xcInd - leftWidth - 2), min(mat.shape[1], xcInd + 2)):
            for y in range(max(0, ycInd - heightDown - 2), min(mat.shape[2], ycInd + 2)):
                xa = x - xcInd
                ya = y - ycInd
                if math.pow(xa, 2)/math.pow(a,2) + math.pow(ya,2)/math.pow(heightDown,2) <= 0.25:
                    mat[instanceInd, x, y] = 1

        a = math.sqrt(math.pow(rightWidth,2) + math.pow(heightUp,2))
        for x in range(max(0, xcInd - 2), min(mat.shape[1], xcInd + rightWidth + 2)):
            for y in range(max(0, ycInd - 2), min(mat.shape[2], ycInd + heightUp + 2)):
                xa = x - xcInd
                ya = y - ycInd
                if math.pow(xa, 2)/math.pow(a,2) + math.pow(ya,2)/math.pow(heightUp,2) <= 0.25:
                    mat[instanceInd, x, y] = 1

        a = math.sqrt(math.pow(rightWidth,2) + math.pow(heightDown,2))
        for x in range(max(0, xcInd - 2), min(mat.shape[1], xcInd + rightWidth + 2)):
            for y in range(max(0, ycInd - heightDown - 2), min(mat.shape[2], ycInd + 2)):
                xa = x - xcInd
                ya = y - ycInd
                if math.pow(xa, 2)/math.pow(a,2) + math.pow(ya,2)/math.pow(heightDown,2) <= 0.25:
                    mat[instanceInd, x, y] = 1
    _getAsymmetricEllipse_kernel = cuda.jit(device=True)(_getAsymmetricEllipse)

    ## peaks: list of arrays with the elements:
    ##        rt, mz,leftRTBorder, rightRTBorder, mzDeviation
    ##
    ## walls: list of arrays with the elements:
    ##        mz, rtstart, rtend, mzDeviation
    ##
    ## backgrounds: list of arrays with the elements:
    ##        rt, mz, mzdev
    ##
    ## Note: walls and backgrounds can be very large, rt center will be picked randomly
    ## Note: actual mz deviation of peaks will be randomized but is at least mzLowerBorder and mzUpperBorder
    def _generateTestExamples(rng_states, mzs, ints, times, peaksCount, peaks, walls, backgrounds, instances, areaRTs, areaMZs, peakTypes, centers, boxes, maxPopulation, intensityScales, randomnessFactor, overlapMinIOU, overlapMaxIOU, overlapRemain):
        supports   = cuda.local.array(shape = mzSlices, dtype=numba.float32)
        dsupports  = cuda.local.array(shape = mzSlices, dtype=numba.float32)
        areaTimes  = cuda.local.array(shape = rtSlices, dtype=numba.float32)
        dareaTimes = cuda.local.array(shape = rtSlices, dtype=numba.float32)
        temp       = cuda.local.array(shape = (1, rtSlices, mzSlices), dtype=numba.float32)

        ## Classes of training instances
        ## Main classes 0.. single isomer, 1.. isomer with left and right overlapping elutions, 2.. isomer with left overlapping elution, 3.. isomer with right overlapping elution, 4.. wall, 5.. background
        rClass = cuda.local.array(shape = (5), dtype=numba.float32)
        rClass[0] = 0; rClass[1] = 1; rClass[2] = 2; rClass[3] = 3;
        rClass[4] = 4;
        ## Distraction classes
        dClass = cuda.local.array(shape = (4), dtype=numba.float32)
        dClass[0] = 0; dClass[1] = 0; dClass[2] = 4; dClass[3] = 4;
        #dClass[4] = 5;

        lInd = cuda.grid(1)
        lLen = cuda.gridsize(1)

        s = 0
        for i in range(times.shape[0]-1):
            s = s + times[i+1]-times[i]
        meanDifferenceScans = s / times.shape[0]

        for instanceInd in range(lInd, instances.shape[0], lLen):
            oRTShift = 0 #xoroshiro128p_uniform_float32(rng_states, lInd) * 2 * rtSlices/16 - (rtSlices/16)
            oMZShift = 0 #xoroshiro128p_uniform_float32(rng_states, lInd) * 5 - 2.5
            populationInd = 0
            tempCPopMax = math.ceil(xoroshiro128p_uniform_float32(rng_states, lInd)*maxPopulation)
            while populationInd < tempCPopMax:
                ind         = -1
                centerRT    =  0
                mzLow       =  0
                mzHigh      =  0
                mpTyp       = -1
                mpRT        =  0
                mpMZ        =  0
                mpRTLeft    =  0
                mpRTRight   =  0
                mpMZLow     =  0
                mpMZHigh    =  0
                mpCentRTInd =  0
                mpCentMZInd =  0
                mpCentInt   =  0
                typR        = -1
                typ         = -1  #[isFullPeak, hasCoelutingPeaksBothSides, hasCoelutingPeakLeft, hasCoelutingPeakRight, isBackground, isWall]

                ## main info in center of the area
                if populationInd == 0:
                    typR = rClass[int(math.floor(xoroshiro128p_uniform_float32(rng_states, lInd) * rClass.shape[0]))]

                    ## Peak (change is higher since it is divided into four subgroups [single, left or right isomer, left and right isomer])
                    if typR <= 3 and peaks.shape[0] > 0:
                        typ       = 0
                        ind       = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(peaks.shape[0]-1))
                        centerRT  = peaks[ind, 0]
                        centerMZ  = peaks[ind, 1]
                        mzDevPPM  = peaks[ind, 4]
                        devMult   = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1

                        mpTyp     = 0
                        mpRT      = peaks[ind, 0]
                        mpMZ      = peaks[ind, 1]
                        mpRTLeft  = peaks[ind, 2]
                        mpRTRight = peaks[ind, 3]
                        mpMZLow   = mpMZ * (1 - peaks[ind, 4] / 6 * 2 / 1E6)
                        mpMZHigh  = mpMZ * (1 + peaks[ind, 4] / 6 * 2 / 1E6)

                    ## Wall
                    elif typR == 4 and walls.shape[0] > 0:
                        typ       = 4
                        ind       = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(walls.shape[0]-1))
                        centerRT  = walls[ind, 1] + (walls[ind, 2]-walls[ind, 1]) * xoroshiro128p_uniform_float32(rng_states, lInd)
                        centerMZ  = walls[ind, 0]
                        mzDevPPM  = walls[ind, 3]
                        devMult   = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1

                        mpTyp     = 4
                        mpRT      = centerRT
                        mpMZ      = walls[ind,0]
                        mpRTLeft  = walls[ind,1]
                        mpRTRight = walls[ind,2]
                        mpMZLow   = mpMZ * (1 - walls[ind, 3] / 2 / 1E6)
                        mpMZHigh  = mpMZ * (1 + walls[ind, 3] / 2 / 1E6)

                    ## Background
                    elif typR == 5 and backgrounds.shape[0] > 0:
                        typ       = 5
                        ind       = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(backgrounds.shape[0]-1))
                        centerRT  = backgrounds[ind, 0]
                        centerMZ  = backgrounds[ind, 1]
                        mzDevPPM  = backgrounds[ind, 2]
                        devMult   = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1

                        mpTyp     = 5
                        mpRT      = backgrounds[ind, 0]
                        mpMZ      = backgrounds[ind, 1]
                        mpRTLeft  = backgrounds[ind, 0] - (xoroshiro128p_uniform_float32(rng_states, lInd)*4+1) * meanDifferenceScans
                        mpRTRight = backgrounds[ind, 0] + (xoroshiro128p_uniform_float32(rng_states, lInd)*4+1) * meanDifferenceScans
                        mpMZLow   = mpMZ * (1 - backgrounds[ind, 2] / 2 / 1E6)
                        mpMZHigh  = mpMZ * (1 + backgrounds[ind, 2] / 2 / 1E6)

                    ## unknown class
                    else:
                        raise RuntimeError("RuntimeError: unknonw class type in generateTestExamples")

                    mzLow     = mpMZ * (1 - mzDevPPM * devMult / 2 / 1E6) * (1 + oMZShift / 1E6)
                    mzHigh    = mpMZ * (1 + mzDevPPM * devMult / 2 / 1E6) * (1 + oMZShift / 1E6)

                ## distraction augmentation not in center of the area
                if populationInd > 0:
                    typR = dClass[int(math.floor(xoroshiro128p_uniform_float32(rng_states, lInd) * dClass.shape[0]))]

                    ## Peak
                    if typR <= 3:
                        sind     = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(peaks.shape[0]-1))
                        rtOffset = meanDifferenceScans * rtSlices/2 * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1)
                        centerRT = peaks[sind, 0] + rtOffset
                        mzOffset = peaks[sind, 1]/1E6 * peaks[sind, 4] * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1) * 2
                        mzDevPPM = peaks[sind, 4] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                        devMult  = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                        mzLow    = (peaks[sind, 1] - mzOffset) * (1 - mzDevPPM * devMult / 2 / 1E6)
                        mzHigh   = (peaks[sind, 1] - mzOffset) * (1 + mzDevPPM * devMult / 2 / 1E6)
                        crtDiff  = mpRT - peaks[sind, 0]

                    ## Wall
                    elif typR == 4:
                        sind     = round(xoroshiro128p_uniform_float32(rng_states, lInd)*walls.shape[0])
                        centerRT = walls[sind, 1] + (walls[sind, 2]-walls[sind, 1]) * xoroshiro128p_uniform_float32(rng_states, lInd)
                        mzOffset = walls[sind, 0]/1E6 * walls[sind, 3] * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1) * 2
                        mzDevPPM = walls[sind, 3] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                        devMult  = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                        mzLow    = (walls[sind, 0] - mzOffset) * (1 - mzDevPPM * devMult / 2 / 1E6)
                        mzHigh   = (walls[sind, 0] - mzOffset) * (1 + mzDevPPM * devMult / 2 / 1E6)

                    ## Background
                    elif typR == 5:
                        sind     = round(xoroshiro128p_uniform_float32(rng_states, lInd)*backgrounds.shape[0])
                        rtOffset = meanDifferenceScans * rtSlices/2 * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1)
                        centerRT = backgrounds[sind, 0] + rtOffset
                        mzOffset = backgrounds[sind, 1]/1E6 * backgrounds[sind, 2] * (xoroshiro128p_uniform_float32(rng_states, lInd) * 2 - 1) * 2
                        mzDevPPM = backgrounds[sind, 2] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                        devMult  = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                        mzLow    = (backgrounds[sind, 1] - mzOffset) * (1 - mzDevPPM * devMult / 2 / 1E6)
                        mzHigh   = (backgrounds[sind, 1] - mzOffset) * (1 + mzDevPPM * devMult / 2 / 1E6)

                    ## unknown class
                    else:
                        raise RuntimeError("RuntimeError: unknonw class type in generateTestExamples")

                ## calculate supports for the addition and save them
                for y in range(mzSlices):
                    supports[y] = mzLow + (mzHigh-mzLow)*y/(mzSlices-1)
                    areaMZs[instanceInd, y] = supports[y]

                ## combine area with previous area
                peakbot.cuda._renovateArea_kernel(mzs, ints, times, peaksCount, centerRT + oRTShift, supports, temp, 0, areaTimes, (supports[1]-supports[0])*8)

                ## get center intensity
                cIndRT = peakbot.cuda.getBestMatch_kernel(areaTimes, mpRT)
                cIndMZ = peakbot.cuda.getBestMatch_kernel(supports , mpMZ)
                maxInt = temp[0, cIndRT, cIndMZ]
                if populationInd == 0:
                    mpCentRTInd = cIndRT
                    mpCentMZInd = cIndMZ
                    mpCentInt = maxInt
                    
                ## scale area
                ## and save area times
                r = 1 + (intensityScales - 1) * xoroshiro128p_uniform_float32(rng_states, lInd)
                if xoroshiro128p_uniform_float32(rng_states, lInd) < 0.5:
                    r = 1/r
                for x in range(rtSlices):
                    if populationInd == 0:
                        areaRTs[instanceInd, x] = areaTimes[x]
                    for y in range(mzSlices):
                        if temp[0,x,y] > 0:
                            temp[0,x,y] = instances[instanceInd,x,y] + (temp[0,x,y] / maxInt * (1 + xoroshiro128p_uniform_float32(rng_states, lInd)*randomnessFactor - randomnessFactor/2) * r)
                ## only use distraction if the center is not too much hidden from it (at least 10 times as intensive)
                if populationInd == 0 or (temp[0,mpCentRTInd,mpCentMZInd]-mpCentInt) < mpCentInt/10:
                    for x in range(rtSlices):
                        for y in range(mzSlices):
                            instances[instanceInd,x,y] = temp[0,x,y]

                ## Add isomeric compounds if the center peak is a chromatographic peak
                if populationInd == 0 and typR <= 3:

                    ## add isomeric peak on the left side
                    if typR == 1 or typR == 2:
                        tries = 40
                        ok    = False
                        intOther = 0
                        while tries > 0 and not ok:
                            tries    = tries - 1
                            sind     = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(peaks.shape[0]-1))
                            rtOffset = meanDifferenceScans * (0.05 + xoroshiro128p_uniform_float32(rng_states, lInd))*rtSlices/3
                            mzDevPPM = peaks[sind, 4] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                            devMult  = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                            mzLow    = peaks[sind, 1] * (1 - mzDevPPM * devMult / 2 / 1E6)
                            mzHigh   = peaks[sind, 1] * (1 + mzDevPPM * devMult / 2 / 1E6)
                            crtDiff  = peaks[ind,0] - peaks[sind,0]

                            if overlapMinIOU <= _iou_kernel(peaks[sind,2]+crtDiff-rtOffset, peaks[sind,3]+crtDiff-rtOffset, 0, 1, peaks[ind,2], peaks[ind,3], 0, 1) <= overlapMaxIOU and \
                                _leftNot_kernel(peaks[sind,2]+crtDiff-rtOffset, peaks[sind,3]+crtDiff-rtOffset, 0, 1, peaks[ind,2], peaks[ind,3], 0, 1) >= overlapRemain and \
                                _rightNot_kernel(peaks[sind,2]+crtDiff-rtOffset, peaks[sind,3]+crtDiff-rtOffset, 0, 1, peaks[ind,2], peaks[ind,3], 0, 1) >= overlapRemain:

                                ## calculate supports for the addition
                                for t in range(mzSlices):
                                    dsupports[t] = 0
                                    dsupports[t] = mzLow + (mzHigh-mzLow)*(t/(mzSlices-1))*(1+oMZShift/1E6)
                                peakbot.cuda._renovateArea_kernel(mzs, ints, times, peaksCount, peaks[sind, 0] + rtOffset + oRTShift, dsupports, temp, 0, dareaTimes, (supports[1]-supports[0])*8)

                                ## get center intensity
                                cIndRT = peakbot.cuda.getBestMatch_kernel(dareaTimes, peaks[sind, 0] + rtOffset + oRTShift)
                                cIndMZ = peakbot.cuda.getBestMatch_kernel(dsupports , (mzHigh+mzLow)/2)
                                intOther = temp[0, cIndRT, cIndMZ]
                                
                                if temp[0, cIndRT, cIndMZ] > 0:
                                    ## scale area
                                    r = 1 + (intensityScales - 1) * xoroshiro128p_uniform_float32(rng_states, lInd)
                                    if xoroshiro128p_uniform_float32(rng_states, lInd) < 0.5:
                                        r = 1/r
                                    for x in range(rtSlices):
                                        for y in range(mzSlices):
                                            temp[0,x,y] = instances[instanceInd, x, y] + (temp[0,x,y] / intOther * (1 + xoroshiro128p_uniform_float32(rng_states, lInd)*randomnessFactor - randomnessFactor/2) * r)

                                    total = -1
                                    newInd = -1
                                    for x in range(rtSlices):
                                        if areaTimes[x] > -1 and peaks[ind, 2] <= areaTimes[x] <= peaks[sind,3]+crtDiff-rtOffset:
                                            s = 0
                                            for y in range(mzSlices):
                                                cmz = supports[y]
                                                if mpMZLow <= cmz <= mpMZHigh:
                                                    s = s + temp[0,x,y]
                                            if total == -1 or s < total:
                                                total = s
                                                newInd = x

                                    if newInd >= 0 and areaTimes[newInd] > mpRTLeft and (temp[0,mpCentRTInd,mpCentMZInd]-mpCentInt) < mpCentInt / 10:
                                        ## update left border and make typ if a leftPeak (or both)
                                        mpRTLeft = areaTimes[newInd]
                                        typ = 2
                                        for x in range(rtSlices):
                                            for y in range(mzSlices):
                                                instances[instanceInd,x,y] = temp[0,x,y]
                                        ok = True

                    ## add isomeric peak on the right side
                    if typR == 1 or typR == 3:
                        tries = 40
                        ok    = False
                        while tries > 0 and not ok:
                            tries    = tries - 1
                            sind     = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(peaks.shape[0]-1))
                            rtOffset = meanDifferenceScans * (0.05 + xoroshiro128p_uniform_float32(rng_states, lInd))*rtSlices/3
                            mzDevPPM = peaks[sind, 4] * (1 + xoroshiro128p_uniform_float32(rng_states, lInd) * 0.5)
                            devMult  = 3 + xoroshiro128p_uniform_float32(rng_states, lInd)*2-1
                            mzLow    = peaks[sind, 1] * (1 - mzDevPPM * devMult / 2 / 1E6)
                            mzHigh   = peaks[sind, 1] * (1 + mzDevPPM * devMult / 2 / 1E6)
                            crtDiff  = peaks[ind,0] - peaks[sind,0]

                            if overlapMinIOU <= _iou_kernel(peaks[ind,2], peaks[ind,3], 0, 1, peaks[sind,2]+crtDiff+rtOffset, peaks[sind,3]+crtDiff+rtOffset, 0, 1) <= overlapMaxIOU and \
                                _rightNot_kernel(peaks[ind,2], peaks[ind,3], 0, 1, peaks[sind,2]+crtDiff+rtOffset, peaks[sind,3]+crtDiff+rtOffset, 0, 1) >= overlapRemain and \
                                _leftNot_kernel(peaks[ind,2], peaks[ind,3], 0, 1, peaks[sind,2]+crtDiff+rtOffset, peaks[sind,3]+crtDiff+rtOffset, 0, 1) >= overlapRemain:

                                ## calculate supports for the addition
                                for t in range(mzSlices):
                                    dsupports[t] = 0
                                    dsupports[t] = mzLow + (mzHigh-mzLow)*(t/(mzSlices-1))*(1+oMZShift/1E6)
                                peakbot.cuda._renovateArea_kernel(mzs, ints, times, peaksCount, peaks[sind, 0] - rtOffset + oRTShift, dsupports, temp, 0, dareaTimes, (supports[1]-supports[0])*8)

                                ## get center intensity
                                cIndRT = peakbot.cuda.getBestMatch_kernel(dareaTimes, peaks[sind, 0] - rtOffset + oRTShift)
                                cIndMZ = peakbot.cuda.getBestMatch_kernel(dsupports , (mzHigh+mzLow)/2)
                                intOther = temp[0, cIndRT, cIndMZ]
                                
                                if temp[0, cIndRT, cIndMZ] > 0:
                                    ## scale area
                                    r = 1 + (intensityScales - 1) * xoroshiro128p_uniform_float32(rng_states, lInd)
                                    if xoroshiro128p_uniform_float32(rng_states, lInd) < 0.5:
                                        r = 1/r
                                    for x in range(rtSlices):
                                        for y in range(mzSlices):
                                            temp[0,x,y] = instances[instanceInd, x, y] + (temp[0,x,y] / intOther * (1 + xoroshiro128p_uniform_float32(rng_states, lInd)*randomnessFactor - randomnessFactor/2) * r)

                                    total = -1
                                    newInd = -1
                                    for x in range(rtSlices):
                                        if areaTimes[x] > -1 and peaks[sind,2]+crtDiff+rtOffset <= areaTimes[x] <= peaks[ind, 3]:
                                            s = 0
                                            for y in range(mzSlices):
                                                cmz = supports[y]
                                                if mpMZLow <= cmz <= mpMZHigh:
                                                    s = s + temp[0, x, y]
                                            if total == -1 or s < total:
                                                total = s
                                                newInd = x

                                    if newInd >= 0 and areaTimes[newInd] < mpRTRight and (temp[0,mpCentRTInd,mpCentMZInd]-mpCentInt) < mpCentInt / 10:
                                        ## update right border and make type a right peak (or both)
                                        mpRTRight = areaTimes[newInd]
                                        ## if there is a left isomer make it a partPeak, otherwise make it a rightPeak
                                        if typ == 0:
                                            typ = 3
                                        elif typ == 2:
                                            typ = 1
                                        for x in range(rtSlices):
                                            for y in range(mzSlices):
                                                instances[instanceInd, x, y] = temp[0,x,y]
                                        ok = True

                ## Update center parameters
                if populationInd == 0:
                    mpTyp = typ
                    peakTypes[instanceInd, typ] = 1

                    if typ < 4:
                        centers[instanceInd, 0] = peakbot.cuda.getBestMatch_kernel(areaTimes, mpRT)
                        centers[instanceInd, 1] = peakbot.cuda.getBestMatch_kernel(supports , mpMZ)
                        boxes[instanceInd, 0]   = peakbot.cuda.getBestMatch_kernel(areaTimes, mpRTLeft)
                        boxes[instanceInd, 1]   = peakbot.cuda.getBestMatch_kernel(areaTimes, mpRTRight)
                        boxes[instanceInd, 2]   = peakbot.cuda.getBestMatch_kernel(supports , mpMZLow)
                        boxes[instanceInd, 3]   = peakbot.cuda.getBestMatch_kernel(supports , mpMZHigh)

                    else:
                        centers[instanceInd, 0] = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(rtSlices))
                        centers[instanceInd, 1] = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(mzSlices))
                        boxes[instanceInd, 0]   = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(rtSlices))
                        boxes[instanceInd, 1]   = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(rtSlices))
                        boxes[instanceInd, 2]   = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(mzSlices))
                        boxes[instanceInd, 3]   = round(xoroshiro128p_uniform_float32(rng_states, lInd)*(mzSlices))
                    
                    if mpTyp == 5 and populationInd > 1:
                        populationInd = 1E5
                
                populationInd = populationInd + 1

            ## Scale to a maximum intensity of 1
            mVal = 0
            for x in range(rtSlices):
                for y in range(mzSlices):
                    mVal = max(mVal, instances[instanceInd, x, y])
            if mVal > 0:
                for x in range(rtSlices):
                     for y in range(mzSlices):
                        instances[instanceInd, x, y] = instances[instanceInd, x, y] / mVal
    generateTestExamples = cuda.jit()(_generateTestExamples)

    rng_states = create_xoroshiro128p_states(griddim * blockdim, seed=2021)
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

    genSamples = blockdim * griddim

    ## local instances
    instances   = np.zeros((genSamples, rtSlices, mzSlices), np.float32)
    areaRTs     = np.zeros((genSamples, rtSlices), np.float32)
    areaMZs     = np.zeros((genSamples, mzSlices), np.float32)
    peakTypes   = np.zeros((genSamples, peakbot.Config.NUMCLASSES), np.float32)
    centers     = np.zeros((genSamples, 2), np.float32)
    boxes       = np.zeros((genSamples, 4), np.float32)

    ## gpu memory instances
    d_instances      = cuda.to_device(instances)
    d_areaRTs        = cuda.to_device(areaRTs)
    d_areaMZs        = cuda.to_device(areaMZs)
    d_peakTypes      = cuda.to_device(peakTypes)
    d_centers        = cuda.to_device(centers)
    d_boxes          = cuda.to_device(boxes)

    if verbose:
        print("  | Generating %d examples in %d batches on the GPU"%(nTestExamples, math.ceil(nTestExamples/batchSize)))
        print("  | .. with the current values of blockdim (%d) and griddim (%d) %d samples will be generated on the GPU"%(blockdim, griddim, genSamples))
        print("  | .. size of the necessary objects is %5.1f Mb (and some more local memory is required by the kernels)"%((instances.nbytes+areaRTs.nbytes+areaMZs.nbytes+peakTypes.nbytes+centers.nbytes+boxes.nbytes)/1E6))
        print("  | .. some more instances than specified by the parameters will be generated. Rounding (ceiling) causes this.")
        print("  | .. Attention: The Windows operating system might fail here. This happens when the GPU calculations exceed 2 seconds (default).")
        print("  | ..            Should this happen, please have a look on the PeakBot homepage (https://github.com/christophuv/PeakBot) to see how to fix it.")
        print("  | .. Attention: Some operating systems (e.g., Windows 10) might be unresponsive for several seconds until this process has completed.")
        print("  | ..            This is normal behaviour since priority is given to the GPU calculations rather than the display update of the operating system.")
        print("  | ")
    outVal = 0
    skipped = 0
    while os.path.exists("%s/%s%d.pickle"%(exportPath, instancePrefix, outVal)):
        outVal = outVal + 1
        skipped = skipped + 1
    if skipped > 0 and verbose:
        print("\r  | .. %6d instance batches already existed. The new ones are appended starting with id %s                           "%(skipped, skipped))

    genClasses = [0,0,0,0,0,0]
    with tqdm.tqdm(total = nTestExamples, desc="  | .. current", unit="instances", smoothing=0, disable=not verbose) as pbar:
        ## generate ahead to support overlapping kernel execution and pickle file dumping
        resetVars[griddim, blockdim](d_instances, d_areaRTs, d_areaMZs, d_peakTypes, d_centers, d_boxes)
        generateTestExamples[griddim, blockdim](rng_states, d_mzs, d_ints, d_times, d_peaksCount, d_peaks, d_walls, d_backgrounds, d_instances, d_areaRTs, d_areaMZs, d_peakTypes, d_centers, d_boxes, maxPopulation, intensityScales, randomnessFactor, overlapMinIOU, overlapMaxIOU, overlapRemain)
        genTestExamples = 0

        while genTestExamples < nTestExamples:

            cuda.synchronize()
            instances  = d_instances.copy_to_host() ;
            areaRTs    = d_areaRTs.copy_to_host()   ; areaMZs = d_areaMZs.copy_to_host();
            peakTypes  = d_peakTypes.copy_to_host() ; centers = d_centers.copy_to_host(); boxes = d_boxes.copy_to_host();

            resetVars[griddim, blockdim](d_instances,  d_areaRTs, d_areaMZs, d_peakTypes, d_centers, d_boxes)
            generateTestExamples[griddim, blockdim](rng_states, d_mzs, d_ints, d_times, d_peaksCount, d_peaks, d_walls, d_backgrounds, d_instances, d_areaRTs, d_areaMZs, d_peakTypes, d_centers, d_boxes, maxPopulation, intensityScales, randomnessFactor, overlapMinIOU, overlapMaxIOU, overlapRemain)

            use = np.logical_and(np.sum(peakTypes, (1)) == 1, np.amax(instances, (1,2)) == 1)
            instances = instances[use,:,:]
            areaRTs   = areaRTs[use, :]
            areaMZs   = areaMZs[use, :]
            peakTypes = peakTypes[use, :]
            centers   = centers[use, :]
            boxes     = boxes[use, :]

            assert all(np.amax(instances, (1,2)) <= 1), "LCHRMSarea is not scaled to a maximum of 1 '%s'"%(str(np.amax(instances, (1,2))))

            cStart = 0
            while cStart < instances.shape[0] and cStart+batchSize < instances.shape[0]:
                a = cStart
                b = cStart + batchSize
                toDump = {"LCHRMSArea"      : instances[a:b,:,:],
                          "areaRTs"         : areaRTs  [a:b,:],
                          "areaMZs"         : areaMZs  [a:b,:],
                          "peakType"        : peakTypes[a:b,:],
                          "center"          : centers  [a:b,:],
                          "box"             : boxes    [a:b,:],
                          "chromatogramFile": [fileIdentifier for i in range(a,b)]}
                pickle.dump(toDump, open(os.path.join(exportPath, "%s%d.pickle"%(instancePrefix, outVal)), "wb"))

                genClasses = [genClasses[0] + sum(peakTypes[a:b,0]==1), genClasses[1] + sum(peakTypes[a:b,1]==1), genClasses[2] + sum(peakTypes[a:b,2]==1),
                              genClasses[3] + sum(peakTypes[a:b,3]==1), genClasses[4] + sum(peakTypes[a:b,4]==1), genClasses[5] + sum(peakTypes[a:b,5]==1)]
                cStart += batchSize
                outVal += 1
                genTestExamples += batchSize
                pbar.update(batchSize)


    cuda.synchronize()

    d_mzs = None; d_ints = None; d_times = None; d_peaksCount = None; d_peaks = None;
    d_walls = None; d_backgrounds = None; d_instances = None; 
    d_areaRTs = None; d_areaMZs = None; d_peakTypes = None; d_centers = None; d_boxes = None;
    cuda.defer_cleanup()

    if verbose:
        print("  | .. generated %d training instances (%d are valid)"%(genTestExamples, sum(genClasses)))
        print("  |    .. %7d (%4.1f%%) single peaks"                        %(genClasses[0], 100*genClasses[0]/genTestExamples))
        print("  |    .. %7d (%4.1f%%) peaks with earlier and later isomers"%(genClasses[1], 100*genClasses[1]/genTestExamples))
        print("  |    .. %7d (%4.1f%%) peaks with earlier isomers"          %(genClasses[2], 100*genClasses[2]/genTestExamples))
        print("  |    .. %7d (%4.1f%%) peaks with later isomers"            %(genClasses[3], 100*genClasses[3]/genTestExamples))
        print("  |    .. %7d (%4.1f%%) walls"                               %(genClasses[4], 100*genClasses[4]/genTestExamples))
        print("  |    .. %7d (%4.1f%%) backgrounds"                         %(genClasses[5], 100*genClasses[5]/genTestExamples))
        print("  | .. took %.1f seconds"%(toc("sampleGeneratingTrainingDataset")))
        print("")
    TabLog().addData(fileIdentifier, "single peaks ", "%d (%.1f%%)"%(genClasses[0], 100*genClasses[0]/genTestExamples))
    TabLog().addData(fileIdentifier, "LeRi iso"     , "%d (%.1f%%)"%(genClasses[1], 100*genClasses[1]/genTestExamples))
    TabLog().addData(fileIdentifier, "Le  iso "     , "%d (%.1f%%)"%(genClasses[2], 100*genClasses[2]/genTestExamples))
    TabLog().addData(fileIdentifier, "Ri iso"       , "%d (%.1f%%)"%(genClasses[3], 100*genClasses[3]/genTestExamples))
    TabLog().addData(fileIdentifier, "walls"        , "%d (%.1f%%)"%(genClasses[4], 100*genClasses[4]/genTestExamples))
    TabLog().addData(fileIdentifier, "backgrounds"  , "%d (%.1f%%)"%(genClasses[5], 100*genClasses[5]/genTestExamples))
    TabLog().addData(fileIdentifier, "Training examples (sec)", toc("sampleGeneratingTrainingDataset"))
