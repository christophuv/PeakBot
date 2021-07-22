import peakbot.core
from peakbot.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary, TabLog

import math
import pickle
import os

import numpy as np
from numba import cuda
import numba
import tqdm

@timeit
def copyToDevice(mzs, ints, times, peaksCount):
    d_mzs = cuda.to_device(mzs)
    d_ints = cuda.to_device(ints)
    d_times = cuda.to_device(times)
    d_peaksCount = cuda.to_device(peaksCount)
    return d_mzs, d_ints, d_times, d_peaksCount

@timeit
def initializeCUDAFunctions(rtSlices = None, mzSlices = None, eicWindowPlusMinus = 30, SavitzkyGolayWindowPlusMinus = 5):
    if rtSlices is None:
        rtSlices = peakbot.Config.RTSLICES
    if mzSlices is None:
        mzSlices = peakbot.Config.MZSLICES
    _extend = 2
    _EICWindow = eicWindowPlusMinus * 2 *_extend + SavitzkyGolayWindowPlusMinus * 2 + 1
    _EICWindowSmoothed = eicWindowPlusMinus * 2 * _extend + 1

    def _findMZGeneric(mzs, ints, times, peaksCount, scanInd, mzleft, mzright):
        pCount=peaksCount[scanInd]    
        if pCount == 0:
            return -1

        min = 0 
        max = pCount

        while min <= max:
            cur = int((max + min) / 2)

            mz = mzs[scanInd, cur]
            if mzleft <= mz and mz <= mzright:
                ## find locally most intense signal
                curmz = mzs[scanInd, cur]
                curint = ints[scanInd, cur]
                while cur-1 > 0 and mzs[scanInd, cur-1] >= mzleft and ints[scanInd, cur-1] > ints[scanInd, cur]:
                    cur -= 1
                while cur+1 < pCount and mzs[scanInd, cur+1] <= mzright and ints[scanInd, cur+1] > ints[scanInd, cur]:
                    cur += 1

                return cur

            if mzs[scanInd, cur] > mzright:
                max = cur - 1
            else:
                min = cur + 1

        return -1
    global _findMZGeneric_kernel
    _findMZGeneric_kernel = cuda.jit(device=True)(_findMZGeneric)

    def _findMostSimilarMZ(mzs, ints, times, peaksCount, scanInd, mzleft, mzright, mzRef):
        pCount=peaksCount[scanInd]    
        if pCount == 0:
            return -1

        min = 0 
        max = pCount

        ## binary search for mz
        while min <= max:
            cur = int((max + min) / 2)

            mz = mzs[scanInd, cur]
            if mzleft <= mz <= mzright:
                ## continue search to the left side
                while cur-1 > 0 and mzleft <= mzs[scanInd, cur-1] and abs(mzRef - mzs[scanInd, cur-1]) < abs(mzRef - mzs[scanInd, cur]):
                    cur -= 1
                ## continue search to the right side
                while cur+1 < pCount and mzs[scanInd, cur+1] <= mzright and abs(mzRef - mzs[scanInd, cur+1]) < abs(mzRef - mzs[scanInd, cur]):
                    cur += 1

                return cur

            if mzs[scanInd, cur] > mzright:
                max = cur - 1
            else:
                min = cur + 1

        return -1
    global _findMostSimilarMZ_kernel
    _findMostSimilarMZ_kernel = cuda.jit(device=True)(_findMostSimilarMZ)

    def _getEIC(mzs, ints, times, peaksCount, searchmz, ppm, eic):
        mzLow = searchmz*(1.-ppm/1E6)
        mzUp = searchmz*(1.+ppm/1E6)

        for scani in range(mzs.shape[0]):
            ind = _findMZGeneric_kernel(mzs, ints, times, peaksCount, scani, mzLow, mzUp)
            if ind < 0:
                ind = 0
            eic[scani] = ind
    global _getEIC_kernel
    _getEIC_kernel = cuda.jit(device=True)(_getEIC)

    def _getEICs(mzs, ints, times, peaksCount, searchmzs, eics):
        lInd = cuda.grid(1)
        lLen = cuda.gridDim.x * cuda.blockDim.x * cuda.gridDim.y * cuda.blockDim.y
        for j in range(lInd, searchmzs.shape[0], lLen):
            _getEIC_kernel(mzs, ints, times, peaksCount, searchmzs[j,:], eics[j,:])
    _getEICs_kernel = None
    _getEICs_kernel = cuda.jit(device=True)(_getEICs)
    global getEICs
    getEICs = cuda.jit()(_getEICs)

    def _getLocalMaxima(mzs, ints, times, peaksCount, maxima, intraScanMaxAdjacentSignalDifferencePPM, interScanMaxSimilarSignalDifferencePPM, minIntensity):
        startX, startY = cuda.grid(2)
        gridX, gridY = cuda.gridsize(2)
        for scani in range(startX, mzs.shape[0], gridX):
            for peaki in range(startY, mzs.shape[1], gridY):
                if peaki < peaksCount[scani] and ints[scani, peaki] >= minIntensity:
                    mz = mzs[scani, peaki]
                    mzDiff = mz*intraScanMaxAdjacentSignalDifferencePPM/1E6

                    mzAdjacentLow = mz*(1.-interScanMaxSimilarSignalDifferencePPM/1E6)
                    mzAdjacentUp  = mz*(1.+interScanMaxSimilarSignalDifferencePPM/1E6)
                    
                    ok = 1
                    
                    if ((peaki-1 > 0 and (ints[scani, peaki-1] < ints[scani, peaki] or (mz-mzs[scani, peaki-1]) > mzDiff)) or peaki == 0) and ((peaki+1 < peaksCount[scani] and (ints[scani, peaki+1] < ints[scani, peaki] or (mzs[scani, peaki+1]-mz) > mzDiff)) or peaki == (peaksCount[scani]-1)):

                        okay = 1 
                        if scani-1 >= 0:
                            pInd = _findMostSimilarMZ_kernel(mzs, ints, times, peaksCount, scani-1, mzAdjacentLow, mzAdjacentUp, mz)
                            if pInd >= 0:
                                if ints[scani-1, pInd] > ints[scani, peaki]:
                                    okay = 0

                                if okay and pInd-1 >= 0 and ints[scani-1, pInd-1] > ints[scani, peaki] and (mzs[scani-1, pInd] - mzs[scani-1, pInd-1]) < mzDiff:
                                    okay = 0

                                if okay and pInd+1 < peaksCount[scani-1] and ints[scani-1, pInd+1] > ints[scani, peaki] and (mzs[scani-1, pInd+1] - mzs[scani-1, pInd]) < mzDiff:
                                    okay = 0

                        if okay and scani+1 < mzs.shape[0]:
                            pInd = _findMostSimilarMZ_kernel(mzs, ints, times, peaksCount, scani+1, mzAdjacentLow, mzAdjacentUp, mz)
                            if pInd >= 0:
                                if ints[scani+1, pInd] > ints[scani, peaki]:
                                    okay = 0

                                if okay and pInd-1 >= 0 and ints[scani+1, pInd-1] > ints[scani, peaki] and (mzs[scani+1, pInd] - mzs[scani+1, pInd-1]) < mzDiff:
                                    okay = 0

                                if okay and pInd+1 < peaksCount[scani+1] and ints[scani+1, pInd+1] > ints[scani, peaki] and (mzs[scani+1, pInd+1] - mzs[scani+1, pInd]) < mzDiff:
                                    okay = 0

                        if okay:
                            maxima[scani, peaki] = 1
    global getLocalMaxima
    getLocalMaxima = cuda.jit()(_getLocalMaxima)

    def _getNormDist(mean, sd, x):
        return 1/math.sqrt(2 * 3.14159265359 * math.pow(sd,2)) * math.exp(-math.pow(x - mean,2) / (2 * math.pow(sd,2)))
    global _getNormDist_kernel
    _getNormDist_kernel = cuda.jit(device=True)(_getNormDist)

    def _getPropsOfSignals(mzs, ints, times, peaksCount, signalProps):
        lInd = cuda.grid(1)
        lLen = cuda.gridsize(1)
        for j in range(lInd, signalProps.shape[0], lLen):
            scanInd = int(signalProps[j,0])
            peakInd = int(signalProps[j,1])
            signalProps[j,2] = times[scanInd]
            signalProps[j,3] = mzs[scanInd, peakInd]
            signalProps[j,4] = ints[scanInd, peakInd]
    global getPropsOfSignals
    getPropsOfSignals = cuda.jit()(_getPropsOfSignals)      

    def _gradientDescendMZProfile(mzs, ints, times, peaksCount, scanInd, peakInd, maxmzDiffPPMAdjacentProfileSignal):
        mz = mzs[scanInd, peakInd]

        ## Gradient descend find borders of mz profile peak
        prevInt = ints[scanInd, peakInd]
        prevMZ = mzs[scanInd, peakInd]
        mzst = prevMZ * prevInt
        totalIntensity = prevInt
        indLow = peakInd
        while indLow-1 >= 0 and ints[scanInd, indLow-1]<prevInt and abs(mzs[scanInd, indLow-1] - prevMZ)*1E6/mz <= maxmzDiffPPMAdjacentProfileSignal:
            indLow = indLow - 1
            mzst = mzst + ints[scanInd, indLow] * mzs[scanInd, indLow]
            totalIntensity = totalIntensity + ints[scanInd, indLow]
            prevInt = ints[scanInd, indLow]
            prevMZ = mzs[scanInd, indLow]

        prevInt = ints[scanInd, peakInd]
        prevMZ = mzs[scanInd, peakInd]
        indUp = peakInd
        while indUp+1 < peaksCount[scanInd] and ints[scanInd, indUp+1]<prevInt and abs(mzs[scanInd, indUp+1] - prevMZ)*1E6/mz <= maxmzDiffPPMAdjacentProfileSignal:
            indUp = indUp + 1
            mzst = mzst + ints[scanInd, indUp] * mzs[scanInd, indUp]
            totalIntensity = totalIntensity + ints[scanInd, indUp]
            prevInt = ints[scanInd, indUp]
            prevMZ = mzs[scanInd, indUp]

        ## calculate weighted mean of mz profile peak
        wMeanMZ = mzst / totalIntensity

        ## calculate weighted ppm stdeviation of mz profile peak
        mzsDev = 0
        for i in range(indLow, indUp+1):
            mzsDev  = mzsDev + ints[scanInd, i] * math.pow(mzs[scanInd, i] - wMeanMZ, 2)
        if indUp == indLow:
            return -1
        wSTDMZ = math.sqrt(mzsDev / ((indUp + 1 - indLow - 1) * totalIntensity / (indUp + 1 - indLow)))

        if wSTDMZ == 0:
            return -1
        scaleFactor =  ints[scanInd, peakInd] / _getNormDist_kernel(wMeanMZ, wSTDMZ, mzs[scanInd, peakInd])

        diffs = 0
        diffsPos = 0
        diffsNeg = 0
        for i in range(indLow, indUp+1):
            hTheo = _getNormDist_kernel(wMeanMZ, wSTDMZ, mzs[scanInd, i]) * scaleFactor
            temp = hTheo - ints[scanInd, i]
            diffs = diffs + math.pow(temp,2)
            if temp > 0:
                diffsPos = diffsPos + math.pow(temp,2)
            else:
                diffsNeg = diffsNeg + math.pow(temp, 2)

        return wSTDMZ * 1E6 / mz * 6
    global _gradientDescendMZProfile_kernel
    _gradientDescendMZProfile_kernel = cuda.jit(device=True)(_gradientDescendMZProfile)

    def _adjacentMZSignalsPPMDifference(mzs, ints, times, peaksCount, signalProps, maxmzDiffPPMAdjacentProfileSignal):
        scanInd = int(signalProps[0])
        peakInd = int(signalProps[1])
        mz = mzs[scanInd, peakInd]
        ppmDiff = signalProps[5]

        ## Gradient descend find borders of mz profile peak
        prevInt = ints[scanInd, peakInd]
        prevMZ = mzs[scanInd, peakInd]
        indLow = peakInd
        while indLow-1 >= 0 and ints[scanInd, indLow-1]<prevInt and abs(mzs[scanInd, indLow-1] - prevMZ)*1E6/mz <= maxmzDiffPPMAdjacentProfileSignal:
            indLow = indLow - 1
            prevInt = ints[scanInd, indLow]
            prevMZ = mzs[scanInd, indLow]

        prevInt = ints[scanInd, peakInd]
        prevMZ = mzs[scanInd, peakInd]
        indUp = peakInd
        while indUp+1 < peaksCount[scanInd] and ints[scanInd, indUp+1]<prevInt and abs(mzs[scanInd, indUp+1] - prevMZ)*1E6/mz <= maxmzDiffPPMAdjacentProfileSignal:
            indUp = indUp + 1
            prevInt = ints[scanInd, indUp]
            prevMZ = mzs[scanInd, indUp]    

        mzsDiffs = 0
        mzsCount = 0
        for i in range(indLow, indUp):
            mzsDiffs = mzsDiffs + (mzs[scanInd, i+1] - mzs[scanInd, i])
            mzsCount = mzsCount + 1

        if mzsCount > 0:
            return mzsDiffs / mzsCount
        else:
            return 0
    global _adjacentMZSignalsPPMDifference_kernel
    _adjacentMZSignalsPPMDifference_kernel = cuda.jit(device=True)(_adjacentMZSignalsPPMDifference)

    def _gradientDescendMZProfileSignals(mzs, ints, times, peaksCount, signalsProps, maxmzDiffPPMAdjacentProfileSignal):
        lInd = cuda.grid(1)
        lLen = cuda.gridsize(1)
        for j in range(lInd, signalsProps.shape[0], lLen):
            bestPPMFit = _gradientDescendMZProfile_kernel(mzs, ints, times, peaksCount, int(signalsProps[j,0]), int(signalsProps[j,1]), maxmzDiffPPMAdjacentProfileSignal)
            signalsProps[j, 5] = bestPPMFit

            ppmDifferenceAdjacentScans = _adjacentMZSignalsPPMDifference_kernel(mzs, ints, times, peaksCount, signalsProps[j,:], maxmzDiffPPMAdjacentProfileSignal)
            signalsProps[j, 6] = ppmDifferenceAdjacentScans
    global gradientDescendMZProfileSignals
    gradientDescendMZProfileSignals = cuda.jit()(_gradientDescendMZProfileSignals)

    def _SavitzkyGolayFilterSmooth(array, smoothed, elements):
        ## array of size n will be smoothed to size elements (n - 2 * SavitzkyGolayWindowPlusMinus)
        ## Filter of window length 2*SavitzkyGolayWindowPlusMinus+1
        ## TODO: Implementation is static, but could be optimized with scipy.savgol_coeffs, which would also make it more flexible in respect to window sizes
        ## r=8; "smoothed[pos] = " + " + ".join(["%f*array[i%+d]"%(c, i-r) for i, c in enumerate(signal.savgol_coeffs(r*2+1, 3, use="dot"))])
        pos = 0
        for i in range(SavitzkyGolayWindowPlusMinus, SavitzkyGolayWindowPlusMinus + elements):
            if SavitzkyGolayWindowPlusMinus == 0:
                smoothed[pos] = array[i]
            if SavitzkyGolayWindowPlusMinus == 1:
                smoothed[pos] = 0.333333*array[i+-1] + 0.333333*array[i+0] + 0.333333*array[i+1]
            if SavitzkyGolayWindowPlusMinus == 2:
                smoothed[pos] = 0.200000*array[i+-2] + 0.200000*array[i+-1] + 0.200000*array[i+0] + 0.200000*array[i+1] + 0.200000*array[i+2]
            if SavitzkyGolayWindowPlusMinus == 3:
                smoothed[pos] = -0.095238*array[i+-3] + 0.142857*array[i+-2] + 0.285714*array[i+-1] + 0.333333*array[i+0] + 0.285714*array[i+1] + 0.142857*array[i+2] + -0.095238*array[i+3]
            if SavitzkyGolayWindowPlusMinus == 4:
                smoothed[pos] = -0.090909*array[i+-4] + 0.060606*array[i+-3] + 0.168831*array[i+-2] + 0.233766*array[i+-1] + 0.255411*array[i+0] + 0.233766*array[i+1] + 0.168831*array[i+2] + 0.060606*array[i+3] + -0.090909*array[i+4]
            if SavitzkyGolayWindowPlusMinus == 5:
                smoothed[pos] = -0.083916*array[i+-5] + 0.020979*array[i+-4] + 0.102564*array[i+-3] + 0.160839*array[i+-2] + 0.195804*array[i+-1] + 0.207459*array[i+0] + 0.195804*array[i+1] + 0.160839*array[i+2] + 0.102564*array[i+3] + 0.020979*array[i+4] + -0.083916*array[i+5]
            if SavitzkyGolayWindowPlusMinus == 6:
                smoothed[pos] = -0.076923*array[i+-6] + 0.000000*array[i+-5] + 0.062937*array[i+-4] + 0.111888*array[i+-3] + 0.146853*array[i+-2] + 0.167832*array[i+-1] + 0.174825*array[i+0] + 0.167832*array[i+1] + 0.146853*array[i+2] + 0.111888*array[i+3] + 0.062937*array[i+4] + -0.000000*array[i+5] + -0.076923*array[i+6]
            if SavitzkyGolayWindowPlusMinus == 7:
                smoothed[pos] = -0.070588*array[i+-7] + -0.011765*array[i+-6] + 0.038009*array[i+-5] + 0.078733*array[i+-4] + 0.110407*array[i+-3] + 0.133032*array[i+-2] + 0.146606*array[i+-1] + 0.151131*array[i+0] + 0.146606*array[i+1] + 0.133032*array[i+2] + 0.110407*array[i+3] + 0.078733*array[i+4] + 0.038009*array[i+5] + -0.011765*array[i+6] + -0.070588*array[i+7]
            if SavitzkyGolayWindowPlusMinus == 8:
                smoothed[pos] = -0.065015*array[i+-8] + -0.018576*array[i+-7] + 0.021672*array[i+-6] + 0.055728*array[i+-5] + 0.083591*array[i+-4] + 0.105263*array[i+-3] + 0.120743*array[i+-2] + 0.130031*array[i+-1] + 0.133127*array[i+0] + 0.130031*array[i+1] + 0.120743*array[i+2] + 0.105263*array[i+3] + 0.083591*array[i+4] + 0.055728*array[i+5] + 0.021672*array[i+6] + -0.018576*array[i+7] + -0.065015*array[i+8]
            pos = pos + 1
    global _SavitzkyGolayFilterSmooth_kernel
    _SavitzkyGolayFilterSmooth_kernel = cuda.jit(device=True)(_SavitzkyGolayFilterSmooth)

    def _gradientDescendRTPeak(mzs, ints, times, peaksCount, signalProps, interScanMaxSimilarSignalDifferencePPM, minWidth, maxWidth, minRatioFactor, minimumIntensity):
        scanInd = int(signalProps[0])
        peakInd = int(signalProps[1])
        mz = signalProps[3]
        
        eic = cuda.local.array(shape = _EICWindow, dtype=numba.float32)
        eicSmoothed = cuda.local.array(shape = _EICWindowSmoothed, dtype=numba.float32)

        pos = 0
        for i in range(scanInd - eicWindowPlusMinus*_extend - SavitzkyGolayWindowPlusMinus, scanInd + eicWindowPlusMinus*_extend + SavitzkyGolayWindowPlusMinus):
            if i >= 0 and i < times.shape[0]:
                ind = _findMostSimilarMZ_kernel(mzs, ints, times, peaksCount, i, mz * (1-interScanMaxSimilarSignalDifferencePPM/1E6), mz * (1+interScanMaxSimilarSignalDifferencePPM/1E6), mz)
                if ind != -1:
                    eic[pos] = ints[i, ind]
                else:
                    eic[pos] = 0
            pos = pos +1

        _SavitzkyGolayFilterSmooth_kernel(eic, eicSmoothed, _EICWindowSmoothed)

        a = int(round(_EICWindowSmoothed/2))
        e = int(round(_EICWindow/2))
        
        ## update apexInd to match derivative
        run = True
        while run:
            if a-1 >=0 and eicSmoothed[a-1] > eicSmoothed[a]:
                a = a - 1
            elif a+1 < _EICWindowSmoothed and eicSmoothed[a+1] > eicSmoothed[a]:
                a = a + 1
            else:
                run = False
        
        ## find left infliction and border scans
        prevDer = 0
        leftOffset = 0
        run = 1
        while run and leftOffset < eicWindowPlusMinus*_extend and (a - leftOffset - 1) >= 0 and (e - leftOffset - 1) >= 0:
            der = eicSmoothed[a - leftOffset - 1] / eicSmoothed[a - leftOffset]
            if der > prevDer and eic[e - leftOffset - 1] > 0:
                leftOffset = leftOffset + 1
                prevDer = der
            else:
                run = 0
        inflLeftOffset = leftOffset 
        infDer = prevDer
        run = 1
        while run and leftOffset < eicWindowPlusMinus*_extend and (a - leftOffset - 1) >= 0 and (e - leftOffset - 1) >= 0:
            if eicSmoothed[a - leftOffset] - eicSmoothed[a - leftOffset - 1] > 0 and eic[e - leftOffset - 1] > 0:
                leftOffset = leftOffset + 1
            else:
                run = 0

        ## find right infliction and border scans
        prevDer = 0
        rightOffset = 0
        run = 1
        while run and rightOffset < eicWindowPlusMinus*_extend and (a + rightOffset + 1) < _EICWindowSmoothed and (e + rightOffset + 1) <= _EICWindow:
            der = eicSmoothed[a + rightOffset + 1] / eicSmoothed[a + rightOffset]
            if der > prevDer and eic[e + rightOffset + 1] > 0:
                rightOffset = rightOffset + 1
                prevDer = der
            else:
                run = 0
        inflRightOffset = rightOffset 
        infDer = prevDer
        run = 1
        while run and rightOffset < eicWindowPlusMinus*_extend and (a + rightOffset + 1) < _EICWindowSmoothed and (e + rightOffset + 1) <= _EICWindow:
            if eicSmoothed[a + rightOffset] - eicSmoothed[a + rightOffset + 1] > 0 and eic[e + rightOffset + 1] > 0:
                rightOffset = rightOffset + 1
            else:
                run = 0

        ## set infliction and border scans as retention times
        signalProps[ 8] = times[min(max(scanInd - leftOffset     , 0), times.shape[0]-1)]
        signalProps[ 9] = times[min(max(scanInd - inflLeftOffset , 0), times.shape[0]-1)]
        signalProps[10] = times[min(max(scanInd + inflRightOffset, 0), times.shape[0]-1)]
        signalProps[11] = times[min(max(scanInd + rightOffset    , 0), times.shape[0]-1)]
        #signalProps[8]=signalProps[9]
        #signalProps[11]=signalProps[10]

        ## calculate and test peak-width and height-ratio (apex to borders)
        peakWidth = rightOffset + leftOffset + 1
        ratio = 0
        infRatioCount = 0
        testInd = scanInd - round(leftOffset * 1.5)
        while testInd < scanInd:
            if testInd >= 0:
                ind = _findMostSimilarMZ_kernel(mzs, ints, times, peaksCount, testInd, mz * (1-interScanMaxSimilarSignalDifferencePPM/1E6), mz * (1+interScanMaxSimilarSignalDifferencePPM/1E6), mz)
                if ind != -1:
                    ratio = max(ratio, ints[scanInd, peakInd]/ints[testInd, ind])
                else:
                    infRatioCount += 1
            testInd = testInd + 1
        
        testInd = scanInd + round(rightOffset * 1.5)
        while testInd > scanInd:
            if testInd < times.shape[0]:
                ind = _findMostSimilarMZ_kernel(mzs, ints, times, peaksCount, testInd, mz * (1-interScanMaxSimilarSignalDifferencePPM/1E6), mz * (1+interScanMaxSimilarSignalDifferencePPM/1E6), mz)
                if ind != -1:
                    ratio = max(ratio, ints[scanInd, peakInd]/ints[testInd, ind])
                else:
                    infRatioCount += 1
            testInd = testInd - 1
            
        signalProps[7] = minWidth <= peakWidth <= maxWidth and ints[scanInd, peakInd] >= minimumIntensity and (ratio >= minRatioFactor or infRatioCount > 3)
    global _gradientDescendRTPeak_kernel
    _gradientDescendRTPeak_kernel = cuda.jit(device=True)(_gradientDescendRTPeak)

    def _gradientDescendRTPeaks(mzs, ints, times, peaksCount, signalsProps, interScanMaxSimilarSignalDifferencePPM, minWidth, maxWidth, minRatioFactor, minimumIntensity):
        lInd = cuda.grid(1)
        lLen = cuda.gridsize(1)
        for j in range(lInd, signalsProps.shape[0], lLen):
            _gradientDescendRTPeak_kernel(mzs, ints, times, peaksCount, signalsProps[j,:], interScanMaxSimilarSignalDifferencePPM, minWidth, maxWidth, minRatioFactor, minimumIntensity)
    global gradientDescendRTPeaks
    gradientDescendRTPeaks = cuda.jit()(_gradientDescendRTPeaks)

    def _joinSplitPeaks(signalsProps):
        lInd = cuda.grid(1)
        lLen = cuda.gridsize(1)
        for j in range(lInd, signalsProps.shape[0], lLen):
            for i in range(signalsProps.shape[0]):
                if abs(signalsProps[i, 3] - signalsProps[j, 3]) * 1E6 / signalsProps[j, 3] < signalsProps[j, 5]/2 and \
                    signalsProps[i, 8] <= signalsProps[j,2] and signalsProps[j,2] <= signalsProps[i,11] and \
                    signalsProps[j, 4] < signalsProps[i, 4]:
                    signalsProps[j,7] = 0
    global joinSplitPeaks
    joinSplitPeaks = cuda.jit()(_joinSplitPeaks)
    
    def _renovateArea(mzs, ints, times, peaksCount, refRT, refSupports, newData, indexToSave, areaTimes, maxOffset):
        
        maxVal = 0 
        
        ## find best matching retention time for the reference
        bestRTMatchDifference = 100000
        bestRTMatchScanInd = -1
        for rtInd in range(0, times.shape[0]):
            rt = times[rtInd]
            if bestRTMatchScanInd == -1 or abs(rt - refRT) < bestRTMatchDifference:
                bestRTMatchDifference = abs(rt - refRT)
                bestRTMatchScanInd = rtInd
        
        for rtOffInd in range(rtSlices):
            rtInd = bestRTMatchScanInd - round(rtSlices / 2) + rtOffInd
            areaTimes[rtOffInd] = -1
            if 0 <= rtInd < mzs.shape[0] and peaksCount[rtInd]>0:
                areaTimes[rtOffInd] = times[rtInd]
            
            for supInd in range(refSupports.shape[0]):
                if 0 <= rtInd < mzs.shape[0] and peaksCount[rtInd]>0:
                    supMZ = refSupports[supInd]
                    newInt = 0

                    leftmzInd = -1
                    rightmzInd = -1

                    bestSuppInd = _findMostSimilarMZ_kernel(mzs, ints, times, peaksCount, rtInd, supMZ - maxOffset, supMZ + maxOffset, supMZ)

                    if bestSuppInd != -1:
                        if supMZ <= mzs[rtInd, bestSuppInd]:
                            if bestSuppInd-1 >= 0 and supMZ - mzs[rtInd, bestSuppInd-1] < maxOffset:
                                leftmzInd = max(0, bestSuppInd-1)
                            rightmzInd = bestSuppInd

                        elif mzs[rtInd, bestSuppInd] < supMZ:
                            leftmzInd = bestSuppInd
                            if bestSuppInd+1 < peaksCount[rtInd] and mzs[rtInd, bestSuppInd+1] - supMZ < maxOffset:
                                rightmzInd = min(peaksCount[rtInd]-1, bestSuppInd + 1)

                        if leftmzInd == -1 and rightmzInd == -1:
                            newInt = 0
                        elif leftmzInd == -1 and rightmzInd != -1:
                            newInt = ints[rtInd, rightmzInd]
                        elif leftmzInd != -1 and rightmzInd == -1:
                            newInt = ints[rtInd, leftmzInd]
                        elif leftmzInd == rightmzInd:
                            newInt = ints[rtInd, leftmzInd]
                        else:
                            newInt = ints[rtInd, leftmzInd] + (ints[rtInd, rightmzInd]-ints[rtInd, leftmzInd])/(mzs[rtInd, rightmzInd]-mzs[rtInd, leftmzInd])*(supMZ-mzs[rtInd, leftmzInd])

                        maxVal = max(maxVal, newInt)
                    
                newData[indexToSave, rtOffInd, supInd] = newInt
                
        return maxVal
    global _renovateArea_kernel
    _renovateArea_kernel = cuda.jit(device=True)(_renovateArea)
    
    def _getBestMatch(vals, val):
        bestMatchInd = -1
        bestMatchDifference = 100000
        for i in range(vals.shape[0]):
            if vals[i] >= 0 and (bestMatchInd == -1 or abs(vals[i] - val) <= bestMatchDifference):
                bestMatchInd = i
                bestMatchDifference = abs(vals[i] - val)
        return bestMatchInd
    global getBestMatch_kernel
    getBestMatch_kernel = cuda.jit(device=True)(_getBestMatch)
    
    def _renovatePeaksToArea(mzs, ints, times, peaksCount, signalsProps, newData, areaRTs, areaMZs, gdProps, fromInd, toInd):
        areaTimes = cuda.local.array(shape = rtSlices, dtype=numba.float32)
        supports = cuda.local.array(shape = mzSlices, dtype=numba.float32)
        
        lInd = cuda.grid(1)
        lLen = cuda.gridsize(1)
        
        for instanceInd in range(lInd, signalsProps.shape[0], lLen):
            if fromInd <= instanceInd < toInd:
                cInstance = instanceInd - fromInd
                ppmdev    = signalsProps[instanceInd, 5]
                refRT     = signalsProps[instanceInd, 2]
                refMZ     = signalsProps[instanceInd, 3]
  
                for y in range(mzSlices):
                    supports[y] = 0
                    supports[y] = refMZ*(1. + (-3 + 6*y/(mzSlices-1))*ppmdev/1E6)
                    areaMZs[cInstance, y] = supports[y]                  
                maxVal = _renovateArea_kernel(mzs, ints, times, peaksCount, refRT, supports, newData, cInstance, areaTimes, (supports[1]-supports[0])*8)
                
                if maxVal > 0:
                    for x in range(rtSlices):
                        areaRTs[cInstance, x] = areaTimes[x]
                        for y in range(mzSlices):
                            newData[cInstance, x, y] /= maxVal                    
                
                gdProps[cInstance, 0] = getBestMatch_kernel(areaTimes, refRT)
                gdProps[cInstance, 1] = getBestMatch_kernel(supports , refMZ)
                gdProps[cInstance, 2] = getBestMatch_kernel(areaTimes, signalsProps[instanceInd,  8])
                gdProps[cInstance, 3] = getBestMatch_kernel(areaTimes, signalsProps[instanceInd, 11])
                gdProps[cInstance, 4] = getBestMatch_kernel(supports , refMZ * (1 - ppmdev/1E6))
                gdProps[cInstance, 5] = getBestMatch_kernel(supports , refMZ * (1 + ppmdev/1E6))
    global renovatePeaksToArea
    renovatePeaksToArea = cuda.jit()(_renovatePeaksToArea)
    
       

@timeit
def preProcessChromatogram(mzxml, fileIdentifier, intraScanMaxAdjacentSignalDifferencePPM, interScanMaxSimilarSignalDifferencePPM, RTpeakWidth, 
                           minApexBorderRatio, minIntensity, exportPath, exportLocalMaxima = "peak-like-shape", 
                           blockdim = (32, 8), griddim = (64, 4),
                           rtSlices = None, mzSlices = None, batchSize = None, 
                           eicWindowPlusMinus = 30, SavitzkyGolayWindowPlusMinus = 3,
                           instancePrefix = None, 
                           verbose = True, verbosePrefix = "", verboseTabLog = False):
    if rtSlices is None:
        rtSlices = peakbot.Config.RTSLICES
    if mzSlices is None:
        mzSlices = peakbot.Config.MZSLICES
    if batchSize is None:
        batchSize = peakbot.Config.BATCHSIZE
    if instancePrefix is None:
        instancePrefix = peakbot.Config.INSTANCEPREFIX
    
    tic("preprocessing")
    if verbose: 
        print(verbosePrefix, "Preprocessing chromatogram with CUDA (GPU-based calculations)", sep="")
        print(verbosePrefix, "  | Parameters", sep="")
        print(verbosePrefix, "  | .. intraScanMaxAdjacentSignalDifferencePPM: ", intraScanMaxAdjacentSignalDifferencePPM, sep="")
        print(verbosePrefix, "  | .. interScanMaxSimilarSignalDifferencePPM: ", interScanMaxSimilarSignalDifferencePPM, sep="")
        print(verbosePrefix, "  | .. RTpeakWidth: ", str(RTpeakWidth), sep="")
        print(verbosePrefix, "  | .. minApexBorderRatio: ", minApexBorderRatio, sep="")
        print(verbosePrefix, "  | .. minIntensity: ", minIntensity, sep="")
        print(verbosePrefix, "  | .. device: ", str(cuda.get_current_device().name), sep="")
        print(verbosePrefix, "  | .. blockdim ", blockdim, ", griddim ", griddim, sep="")
        print(verbosePrefix, "  |", sep="")

    initializeCUDAFunctions(rtSlices, mzSlices, eicWindowPlusMinus, SavitzkyGolayWindowPlusMinus)
    cuda.synchronize()
    
    if verbose:
        print(verbosePrefix, "  | Converted to numpy objects", sep="")
    tic()
    mzs, ints, times, peaksCount = mzxml.convertChromatogramToNumpyObjects(verbose = verbose, verbosePrefix = verbosePrefix)
    if verbose:
        print(verbosePrefix, "  | .. size of objects is %5.1f Mb"%(mzs.nbytes/1E6 + ints.nbytes/1E6 + times.nbytes/1E6 + peaksCount.nbytes/1E6), sep="")
        print(verbosePrefix, "  | .. Attention: size will grow and depend on the number of local maxima in the chromatogram", sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "Conv NP (sec)", toc())

    tic()
    d_mzs, d_ints, d_times, d_peaksCount = copyToDevice(mzs, ints, times, peaksCount)
    cuda.synchronize()
    maxima = np.zeros(mzs.shape, dtype=bool)
    d_maxima = cuda.to_device(maxima)
    if verbose:
        print(verbosePrefix, "  | Copied numpy objects to device memory", sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "CP GPU (sec)", toc())

    tic()
    d_res = getLocalMaxima[griddim, blockdim](d_mzs, d_ints, d_times, d_peaksCount, d_maxima, intraScanMaxAdjacentSignalDifferencePPM, interScanMaxSimilarSignalDifferencePPM, minIntensity)
    cuda.synchronize()
    maxima = d_maxima.copy_to_host()
    if verbose:
        print(verbosePrefix, "  | Found %d local maxima"%(np.sum(maxima)), sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "n locMax", np.sum(maxima))
        TabLog().addData(fileIdentifier, "locMax (sec)", toc())

    tic()
    maximaPropsAll = np.argwhere(maxima > 0)
    d_maxima = None
    ##       0       1       2   3   4          5                6                         7       8             9          10          11
    ## cols: ind-RT, ind-MZ, RT, MZ, intensity, PPMdevMZProfile, ppmDifferenceAdjacentMZs, isPeak, leftRTBorder, infLeftRT, infRightRT, rightRTBorder
    maximaPropsAll = np.column_stack((maximaPropsAll, np.zeros((maximaPropsAll.shape[0], 20), np.float32)))
    maximaPropsAll = maximaPropsAll.astype(np.float32)
    d_maximaPropsAll = cuda.to_device(maximaPropsAll)
    getPropsOfSignals[griddim, blockdim](d_mzs, d_ints, d_times, d_peaksCount, d_maximaPropsAll)
    cuda.synchronize()
    if verbose:
        print(verbosePrefix, "  | Calculated properties of local maxima", sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "prop locMax (sec)", toc())

    tic()
    gradientDescendMZProfileSignals[griddim, blockdim](d_mzs, d_ints, d_times, d_peaksCount, d_maximaPropsAll, intraScanMaxAdjacentSignalDifferencePPM)
    cuda.synchronize()
    maximaPropsAll = d_maximaPropsAll.copy_to_host()
    maximaProps = maximaPropsAll[maximaPropsAll[:,5]>0,:]
    d_maximaProps = cuda.to_device(maximaProps)
    if verbose:
        print(verbosePrefix, "  | Calculated mz peak deviations", sep="")
        print(verbosePrefix, "  | .. there are %d local maxima with an mz profile peak (%.3f%% of all signals)"%(maximaProps.shape[0], 100.*maximaProps.shape[0]/(maxima.shape[0]*maxima.shape[1])), sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "n locMax mzPeak", maximaProps.shape[0])
        TabLog().addData(fileIdentifier, "locMax mzPeak (sec)", toc())

    tic()
    gradientDescendRTPeaks[griddim, blockdim](d_mzs, d_ints, d_times, d_peaksCount, d_maximaProps, interScanMaxSimilarSignalDifferencePPM, RTpeakWidth[0], RTpeakWidth[1], minApexBorderRatio, minIntensity)
    cuda.synchronize()
    peaks = d_maximaProps.copy_to_host()
    peaks = peaks[peaks[:,7]>0,:]
    d_peaks = cuda.to_device(peaks)
    if verbose:
        print(verbosePrefix, "  | Calculated rt peak borders", sep="")
        print(verbosePrefix, "  | .. there are %d local maxima with an mz profile peak and an rt-peak-like shape (%.3f%% of all signals)"%(peaks.shape[0], 100.*maximaProps.shape[0]/(maxima.shape[0]*maxima.shape[1])), sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "n peaks", peaks.shape[0])
        TabLog().addData(fileIdentifier, "peaks (sec)", toc())

    tic()
    joinSplitPeaks[griddim, blockdim](d_peaks)
    cuda.synchronize()
    peaks = d_peaks.copy_to_host()
    peaks = peaks[peaks[:,7]>0,:]
    d_peaks = cuda.to_device(peaks)
    if verbose:
        print(verbosePrefix, "  | Joined noisy local maxima", sep="")
        print(verbosePrefix, "  | .. there are %d local maxima with an mz profile peak and an rt-peak-like shape that are not split (%.3f%% of all signals)"%(peaks.shape[0], 100.*peaks.shape[0]/(maxima.shape[0]*maxima.shape[1])), sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "n join peaks", peaks.shape[0])
        TabLog().addData(fileIdentifier, "join peaks (sec)", toc())
        
    tic()
    cuda.defer_cleanup()
    newData = np.zeros((batchSize, rtSlices, mzSlices), np.float32)
    areaRTs = np.zeros((batchSize, rtSlices), np.float32)
    areaMZs = np.zeros((batchSize, mzSlices), np.float32)
    gdProps = np.zeros((batchSize, 6), np.float32) ## gdProps: apexRTInd, apexMZInd, rtLeftBorderInd, rtRightBorderInd, mzLowestInd, mzHighestInd
    outVal = 0
    
    if exportLocalMaxima is not None:
    
        locMax = None
        d_locMax = None
        if exportLocalMaxima == "peak-like-shape":
            locMax = peaks
            d_locMax = d_peaks
        elif exportLocalMaxima == "localMaxima-with-mzProfile":
            locMax = maximaProps
            d_locMax = d_maximaProps
        elif exportLocalMaxima == "all":
            locMax = maximaPropsAll
            d_locMax = d_maximaPropsAll
        
        if verbose:
            print(verbosePrefix, "  | Exporting %d local maxima to standardized area"%(locMax.shape[0]), sep="")
            print(verbosePrefix, "  | .. %d batch files, %d features per batch each with approximately %5.1f Mb"%(math.ceil(locMax.shape[0]/batchSize), batchSize, newData.nbytes/1E6), sep="")
        
        
        if exportPath is not None:
            if verbose: 
                print(verbosePrefix, "  | .. directory '%s'"%(exportPath), sep="")

            cur = 0
            with tqdm.tqdm(total = math.ceil(locMax.shape[0]/batchSize), desc="  | .. exporting", disable=not verbose) as t:
                while cur < locMax.shape[0]:
                    newData.fill(0)
                    areaRTs.fill(-10) 
                    areaMZs.fill(0)
                    gdProps.fill(0)
                    d_newData = cuda.to_device(newData)
                    d_areaRTs = cuda.to_device(areaRTs)
                    d_areaMZs = cuda.to_device(areaMZs)
                    d_gdProps = cuda.to_device(gdProps)

                    ## TODO: Export leaves a lot of GPU resources unused since only batchSize local maxima are processed at a time.
                    ##       There is much room for improvement. The implementation is similar to the training-export
                    renovatePeaksToArea[griddim, blockdim](d_mzs, d_ints, d_times, d_peaksCount, d_locMax, d_newData, d_areaRTs, d_areaMZs, d_gdProps, cur, cur+batchSize)
                    cuda.synchronize()

                    newData = d_newData.copy_to_host()
                    areaRTs = d_areaRTs.copy_to_host()
                    areaMZs = d_areaMZs.copy_to_host()
                    gdProps = d_gdProps.copy_to_host()

                    use = areaRTs[:,0] > -10
                    pickle.dump({"lcmsArea": newData[use,:,:], 
                                 "areaRTs" : areaRTs[use,:], 
                                 "areaMZs" : areaMZs[use,:],
                                 "gdProps" : gdProps[use, :]}, 
                                open(os.path.join(exportPath, "%s%d.pickle"%(instancePrefix, outVal)), "wb"))

                    cur += batchSize
                    outVal += 1
                    t.update()
                    
        else:
            print("Streaming not yet implemented: TODO")
            import sys
            sys.exit(0)
    
    
        if verbose:
            print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
            print(verbosePrefix, "  |", sep="")
        if verboseTabLog:
            TabLog().addData(fileIdentifier, "export (sec)", toc())

    tic()
    d_mzs            = None
    d_ints           = None
    d_times          = None
    d_peaksCount     = None
    d_maximaPropsAll = None
    d_maximaProps    = None
    d_peaks          = None
    d_newData        = None
    d_areaRTs        = None
    d_areaMZs        = None
    cuda.defer_cleanup()
    
    if verbose: 
        print(verbosePrefix, "  | Pre-processing with CUDA took %.1f seconds"%toc("preprocessing"), sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "preprocessing (sec)", toc("preprocessing"))
        
    return peaks, maximaProps, maximaPropsAll
