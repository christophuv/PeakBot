import peakbot.core
from peakbot.core import tic, toc,timeit, TabLog

import math
import pickle
import os

import numpy as np
from numba import prange
import numba
import tqdm


@timeit
def initializeCPUFunctions(gdMZminRatio = 0.1, gdRTminRatio = 0.01, eicWindowPlusMinus = 30, SavitzkyGolayWindowPlusMinus = 5, numbaParallel = False):
    rtSlices = peakbot.Config.RTSLICES
    mzSlices = peakbot.Config.MZSLICES
    _extend = 2
    _EICWindow = eicWindowPlusMinus * 2 *_extend + SavitzkyGolayWindowPlusMinus * 2 + 1
    _EICWindowSmoothed = eicWindowPlusMinus * 2 * _extend + 1

    global _findMZGeneric
    @numba.jit(nopython=True)
    def _findMZGeneric(mzs, ints, times, peaksCount, scanInd, mzleft, mzright):
        pCount=peaksCount[scanInd]
        if pCount == 0:
            return -1

        min = 0
        max = pCount

        while min <= max:
            cur = int((max + min) / 2)

            if mzleft <= mzs[scanInd, cur] <= mzright:
                ## continue search to the left side
                while cur-1 >= 0 and mzs[scanInd, cur-1] >= mzleft and ints[scanInd, cur-1] > ints[scanInd, cur]:
                    cur = cur - 1
                ## continue search to the right side
                while cur+1 < pCount and mzs[scanInd, cur+1] <= mzright and ints[scanInd, cur+1] > ints[scanInd, cur]:
                    cur = cur + 1

                return cur

            if mzs[scanInd, cur] > mzright:
                max = cur - 1
            else:
                min = cur + 1

        return -1
    
    global _findMostSimilarMZ
    @numba.jit(nopython=True)
    def _findMostSimilarMZ(mzs, ints, times, peaksCount, scanInd, mzleft, mzright, mzRef):
        pCount=peaksCount[scanInd]
        if pCount == 0:
            return -1

        min = 0
        max = pCount

        ## binary search for mz
        while min <= max:
            cur = int((max + min) / 2)

            if mzleft <= mzs[scanInd, cur] <= mzright:
                ## continue search to the left side
                while cur-1 >= 0 and mzs[scanInd, cur-1] >= mzleft and abs(mzRef - mzs[scanInd, cur-1]) < abs(mzRef - mzs[scanInd, cur]):
                    cur = cur - 1
                ## continue search to the right side
                while cur+1 < pCount and mzs[scanInd, cur+1] <= mzright and abs(mzRef - mzs[scanInd, cur+1]) < abs(mzRef - mzs[scanInd, cur]):
                    cur = cur + 1

                return cur

            if mzs[scanInd, cur] > mzright:
                max = cur - 1
            else:
                min = cur + 1

        return -1

    global _getEIC
    @numba.jit(nopython=True)
    def _getEIC(mzs, ints, times, peaksCount, searchmz, ppm, eic):
        mzLow = searchmz*(1.-ppm/1E6)
        mzUp = searchmz*(1.+ppm/1E6)

        for scani in range(mzs.shape[0]):
            ind = _findMZGeneric(mzs, ints, times, peaksCount, scani, mzLow, mzUp)
            if ind < 0:
                ind = 0
            eic[scani] = ind
            
    global _getEICs
    @numba.jit(nopython=True, parallel=numbaParallel)
    def _getEICs(mzs, ints, times, peaksCount, searchmzs, eics):
        for j in prange(searchmzs.shape[0]):
            _getEIC(mzs, ints, times, peaksCount, searchmzs[j,:], eics[j,:])
       
    global _getLocalMaxima     
    @numba.jit(nopython=True, parallel=numbaParallel)
    def _getLocalMaxima(mzs, ints, times, peaksCount, maxima, intraScanMaxAdjacentSignalDifferencePPM, interScanMaxSimilarSignalDifferencePPM, minIntensity):
        for scani in prange(mzs.shape[0]):
            for peaki in range(mzs.shape[1]):
                if peaki < peaksCount[scani] and ints[scani, peaki] >= minIntensity:
                    mz = mzs[scani, peaki]
                    mzDiff = mz*intraScanMaxAdjacentSignalDifferencePPM/1E6

                    mzAdjacentLow = mz*(1.-interScanMaxSimilarSignalDifferencePPM/1E6)
                    mzAdjacentUp  = mz*(1.+interScanMaxSimilarSignalDifferencePPM/1E6)

                    ok = 1

                    if ((peaki-1 > 0 and (ints[scani, peaki-1] < ints[scani, peaki] or (mz-mzs[scani, peaki-1]) > mzDiff)) or peaki == 0) and ((peaki+1 < peaksCount[scani] and (ints[scani, peaki+1] < ints[scani, peaki] or (mzs[scani, peaki+1]-mz) > mzDiff)) or peaki == (peaksCount[scani]-1)):

                        okay = 1
                        if scani-1 >= 0:
                            pInd = _findMostSimilarMZ(mzs, ints, times, peaksCount, scani-1, mzAdjacentLow, mzAdjacentUp, mz)
                            if pInd >= 0:
                                if ints[scani-1, pInd] > ints[scani, peaki]:
                                    okay = 0

                                if okay and pInd-1 >= 0 and ints[scani-1, pInd-1] > ints[scani, peaki] and (mzs[scani-1, pInd] - mzs[scani-1, pInd-1]) < mzDiff:
                                    okay = 0

                                if okay and pInd+1 < peaksCount[scani-1] and ints[scani-1, pInd+1] > ints[scani, peaki] and (mzs[scani-1, pInd+1] - mzs[scani-1, pInd]) < mzDiff:
                                    okay = 0

                        if okay and scani+1 < mzs.shape[0]:
                            pInd = _findMostSimilarMZ(mzs, ints, times, peaksCount, scani+1, mzAdjacentLow, mzAdjacentUp, mz)
                            if pInd >= 0:
                                if ints[scani+1, pInd] > ints[scani, peaki]:
                                    okay = 0

                                if okay and pInd-1 >= 0 and ints[scani+1, pInd-1] > ints[scani, peaki] and (mzs[scani+1, pInd] - mzs[scani+1, pInd-1]) < mzDiff:
                                    okay = 0

                                if okay and pInd+1 < peaksCount[scani+1] and ints[scani+1, pInd+1] > ints[scani, peaki] and (mzs[scani+1, pInd+1] - mzs[scani+1, pInd]) < mzDiff:
                                    okay = 0

                        if okay:
                            maxima[scani, peaki] = 1

    global _getNormDist
    @numba.jit(nopython=True)
    def _getNormDist(mean, sd, x):
        return 1/math.sqrt(2 * 3.14159265359 * math.pow(sd,2)) * math.exp(-math.pow(x - mean,2) / (2 * math.pow(sd,2)))
    
    global _getPropsOfSignals
    @numba.jit(nopython=True, parallel=numbaParallel)
    def _getPropsOfSignals(mzs, ints, times, peaksCount, signalProps):
        for j in prange(signalProps.shape[0]):
            scanInd = int(signalProps[j,0])
            peakInd = int(signalProps[j,1])
            signalProps[j,2] = times[scanInd]
            signalProps[j,3] = mzs[scanInd, peakInd]
            signalProps[j,4] = ints[scanInd, peakInd]
      
    global _gradientDescendMZProfile      
    @numba.jit(nopython=True)
    def _gradientDescendMZProfile(mzs, ints, times, peaksCount, scanInd, peakInd, maxmzDiffPPMAdjacentProfileSignal):
        mz = mzs[scanInd, peakInd]
        intA = ints[scanInd, peakInd]

        ## Gradient descend find borders of mz profile peak
        prevInt = ints[scanInd, peakInd]
        prevMZ = mzs[scanInd, peakInd]
        mzst = prevMZ * prevInt
        totalIntensity = prevInt
        indLow = peakInd
        while indLow-1 >= 0 and ints[scanInd, indLow-1]<prevInt and ints[scanInd, indLow-1]/intA >= gdMZminRatio and abs(mzs[scanInd, indLow-1] - prevMZ)*1E6/mz <= maxmzDiffPPMAdjacentProfileSignal:
            indLow = indLow - 1
            mzst = mzst + ints[scanInd, indLow] * mzs[scanInd, indLow]
            totalIntensity = totalIntensity + ints[scanInd, indLow]
            prevInt = ints[scanInd, indLow]
            prevMZ = mzs[scanInd, indLow]

        prevInt = ints[scanInd, peakInd]
        prevMZ = mzs[scanInd, peakInd]
        indUp = peakInd
        while indUp+1 < peaksCount[scanInd] and ints[scanInd, indUp+1]<prevInt and ints[scanInd, indUp+1]/intA >= gdMZminRatio and abs(mzs[scanInd, indUp+1] - prevMZ)*1E6/mz <= maxmzDiffPPMAdjacentProfileSignal:
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
            return -1, -1
        wSTDMZ = math.sqrt(mzsDev / ((indUp + 1 - indLow - 1) / (indUp + 1 - indLow) * totalIntensity))

        if wSTDMZ == 0:
            return -1, -1

        return wMeanMZ, wSTDMZ * 1E6 / wMeanMZ * 6
    
    global _adjacentMZSignalsPPMDifference
    @numba.jit(nopython=True)
    def _adjacentMZSignalsPPMDifference(mzs, ints, times, peaksCount, signalProps, maxmzDiffPPMAdjacentProfileSignal):
        scanInd = int(signalProps[0])
        peakInd = int(signalProps[1])
        mz = mzs[scanInd, peakInd]
        inte = ints[scanInd, peakInd]
        ppmDiff = signalProps[5]

        ## Gradient descend find borders of mz profile peak
        prevInt = ints[scanInd, peakInd]
        prevMZ = mzs[scanInd, peakInd]
        indLow = peakInd
        while indLow-1 >= 0 and abs(mzs[scanInd, indLow-1] - prevMZ)*1E6/mz <= maxmzDiffPPMAdjacentProfileSignal and ints[scanInd, indLow-1]/inte >= gdMZminRatio and ints[scanInd, indLow-1]<prevInt:
            indLow = indLow - 1
            prevInt = ints[scanInd, indLow]
            prevMZ = mzs[scanInd, indLow]

        prevInt = ints[scanInd, peakInd]
        prevMZ = mzs[scanInd, peakInd]
        indUp = peakInd
        while indUp+1 < peaksCount[scanInd] and abs(mzs[scanInd, indUp+1] - prevMZ)*1E6/mz <= maxmzDiffPPMAdjacentProfileSignal and ints[scanInd, indUp+1]/inte >= gdMZminRatio and ints[scanInd, indUp+1]<prevInt:
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
        
    global _gradientDescendMZProfileSignals
    @numba.jit(nopython=True, parallel=numbaParallel)
    def _gradientDescendMZProfileSignals(mzs, ints, times, peaksCount, signalsProps, maxmzDiffPPMAdjacentProfileSignal):
        for j in prange(signalsProps.shape[0]):
            wMeanMZ, bestPPMFit = _gradientDescendMZProfile(mzs, ints, times, peaksCount, int(signalsProps[j,0]), int(signalsProps[j,1]), maxmzDiffPPMAdjacentProfileSignal)
            signalsProps[j, 3] = wMeanMZ
            signalsProps[j, 5] = bestPPMFit

            ppmDifferenceAdjacentScans = _adjacentMZSignalsPPMDifference(mzs, ints, times, peaksCount, signalsProps[j,:], maxmzDiffPPMAdjacentProfileSignal)
            signalsProps[j, 6] = ppmDifferenceAdjacentScans
            
    global _SavitzkyGolayFilterSmooth
    @numba.jit(nopython=True)
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
                smoothed[pos] = -0.085714*array[i-2] + 0.342857*array[i-1] + 0.485714*array[i+0] + 0.342857*array[i+1] + -0.085714*array[i+2]
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
            
    global _gradientDescendRTPeak
    @numba.jit(nopython=True)
    def _gradientDescendRTPeak(mzs, ints, times, peaksCount, signalProps, signalInd, interScanMaxSimilarSignalDifferencePPM, minWidth, maxWidth, minRatioFactor, minimumIntensity):
        scanInd = int(signalProps[signalInd, 0])
        peakInd = int(signalProps[signalInd, 1])

        ## the mz value of a profile-signal is used rather than the weighted mz value as this can cause problems during eic generation
        mz = mzs[scanInd, peakInd]
        eic = np.zeros(shape = _EICWindow, dtype=numba.float32)
        eicSmoothed = np.zeros(shape = _EICWindowSmoothed, dtype=numba.float32)

        pos = int(round(_EICWindow/2))
        prevMZ = mz
        for i in range(0, int(round((eicWindowPlusMinus*_extend + SavitzkyGolayWindowPlusMinus)/2))):
            cScanInd = scanInd - i
            if 0 <= cScanInd < times.shape[0]:
                ind = _findMostSimilarMZ(mzs, ints, times, peaksCount, cScanInd, prevMZ * (1-interScanMaxSimilarSignalDifferencePPM/1E6), prevMZ * (1+interScanMaxSimilarSignalDifferencePPM/1E6), prevMZ)
                #ind = _findMZGeneric_kernel(mzs, ints, times, peaksCount, cScanInd, prevMZ * (1-interScanMaxSimilarSignalDifferencePPM/1E6), prevMZ * (1+interScanMaxSimilarSignalDifferencePPM/1E6))
                if ind != -1 and pos >= 0:
                    eic[pos] = ints[cScanInd, ind]
                    prevMZ = mzs[cScanInd, ind]
                else:
                    break
            pos = pos - 1

        pos = int(round(_EICWindow/2)) + 1
        prevMZ = mz
        for i in range(1, int(round((eicWindowPlusMinus*_extend + SavitzkyGolayWindowPlusMinus)/2)+1)):
            cScanInd = scanInd + i
            if 0 <= cScanInd < times.shape[0]:
                ind = _findMostSimilarMZ(mzs, ints, times, peaksCount, cScanInd, prevMZ * (1-interScanMaxSimilarSignalDifferencePPM/1E6), prevMZ * (1+interScanMaxSimilarSignalDifferencePPM/1E6), prevMZ)
                #ind = _findMZGeneric_kernel(mzs, ints, times, peaksCount, cScanInd, prevMZ * (1-interScanMaxSimilarSignalDifferencePPM/1E6), prevMZ * (1+interScanMaxSimilarSignalDifferencePPM/1E6))
                if ind != -1 and pos <_EICWindow:
                    eic[pos] = ints[cScanInd, ind]
                    prevMZ = mzs[cScanInd, ind]
                else:
                    break
            pos = pos + 1


        _SavitzkyGolayFilterSmooth(eic, eicSmoothed, _EICWindowSmoothed)
        
        a = int(round(_EICWindowSmoothed/2))
        e = int(round(_EICWindow/2))
        ## update apexInd to match derivative
        run = True
        while run:
            if a-1 >= 0 and eicSmoothed[a-1] > eicSmoothed[a]:
                a = a - 1
            elif a+1 < _EICWindowSmoothed and eicSmoothed[a+1] > eicSmoothed[a]:
                a = a + 1
            else:
                run = False
        run = True
        while run:
            if e-1 >= 0 and eic[e-1] > eic[e]:
                e = e - 1
            elif e+1 < _EICWindow and eic[e+1] > eic[e]:
                e = e + 1
            else:
                run = False

        ## find left infliction and border scans
        prevDer = 0
        leftOffset = 0
        run = 1
        while run and leftOffset+1 < eicWindowPlusMinus*_extend and (a - leftOffset - 1) >= 0 and (e - leftOffset - 1) >= 0:
            if eicSmoothed[a - leftOffset] > 0:
                der = eicSmoothed[a - leftOffset - 1] / eicSmoothed[a - leftOffset]
            else:
                der = 0
            if eic[e - leftOffset - 1] > 0 and der > prevDer:
                leftOffset = leftOffset + 1
                prevDer = der
            else:
                run = 0
        inflLeftOffset = leftOffset
        run = 1
        while run and leftOffset+1 < eicWindowPlusMinus*_extend and (a - leftOffset - 1) >= 0 and (e - leftOffset - 1) >= 0:
            if eic[e - leftOffset - 1] > 0 and eicSmoothed[a] > 0 and eicSmoothed[a - leftOffset - 1] / eicSmoothed[a] >= gdRTminRatio and eicSmoothed[a - leftOffset - 1] < eicSmoothed[a - leftOffset]:
                leftOffset = leftOffset + 1
            else:
                run = 0

        ## find right infliction and border scans
        prevDer = 0
        rightOffset = 0
        run = 1
        while run and rightOffset+1 < eicWindowPlusMinus*_extend and (a + rightOffset + 1) < _EICWindowSmoothed and (e + rightOffset + 1) <= _EICWindow:
            if eicSmoothed[a + rightOffset] > 0:
                der = eicSmoothed[a + rightOffset + 1] / eicSmoothed[a + rightOffset]
            else:
                der = 0
            if eic[e + rightOffset + 1] > 0 and der > prevDer:
                rightOffset = rightOffset + 1
                prevDer = der
            else:
                run = 0
        inflRightOffset = rightOffset
        run = 1
        while run and rightOffset+1 < eicWindowPlusMinus*_extend and (a + rightOffset + 1) < _EICWindowSmoothed and (e + rightOffset + 1) <= _EICWindow:
            if eic[e + rightOffset + 1] > 0 and eicSmoothed[a] > 0 and eicSmoothed[a + rightOffset + 1] / eicSmoothed[a] >= gdRTminRatio and eicSmoothed[a + rightOffset + 1] < eicSmoothed[a + rightOffset]:
                rightOffset = rightOffset + 1
            else:
                run = 0

            ## set infliction and border scans as retention times
            signalProps[signalInd,  8] = times[min(max(scanInd - leftOffset     , 0), times.shape[0]-1)]
            signalProps[signalInd,  9] = times[min(max(scanInd - inflLeftOffset , 0), times.shape[0]-1)]
            signalProps[signalInd, 10] = times[min(max(scanInd + inflRightOffset, 0), times.shape[0]-1)]
            signalProps[signalInd, 11] = times[min(max(scanInd + rightOffset    , 0), times.shape[0]-1)]

            ## calculate and test peak-width and height-ratio (apex to borders)
            peakWidth = rightOffset + leftOffset + 1
            ratio = 0
            infRatioCount = 0
            testInd = scanInd - round(leftOffset * 1.5)
            while testInd < scanInd:
                if testInd >= 0:
                    ind = _findMostSimilarMZ(mzs, ints, times, peaksCount, testInd, mz * (1-interScanMaxSimilarSignalDifferencePPM/1E6), mz * (1+interScanMaxSimilarSignalDifferencePPM/1E6), mz)
                    if ind != -1:
                        ratio = max(ratio, ints[scanInd, peakInd]/ints[testInd, ind])
                    else:
                        infRatioCount += 1
                testInd = testInd + 1

            testInd = scanInd + round(rightOffset * 1.5)
            while testInd > scanInd:
                if testInd < times.shape[0]:
                    ind = _findMostSimilarMZ(mzs, ints, times, peaksCount, testInd, mz * (1-interScanMaxSimilarSignalDifferencePPM/1E6), mz * (1+interScanMaxSimilarSignalDifferencePPM/1E6), mz)
                    if ind != -1:
                        ratio = max(ratio, ints[scanInd, peakInd]/ints[testInd, ind])
                    else:
                        infRatioCount += 1
                testInd = testInd - 1
            
            signalProps[signalInd, 7] = minWidth <= peakWidth <= maxWidth and ints[scanInd, peakInd] >= minimumIntensity and (ratio >= minRatioFactor or infRatioCount > 1)
        
    global _gradientDescendRTPeaks
    @numba.jit(nopython=True, parallel=numbaParallel)
    def _gradientDescendRTPeaks(mzs, ints, times, peaksCount, signalsProps, interScanMaxSimilarSignalDifferencePPM, minWidth, maxWidth, minRatioFactor, minimumIntensity):
        for j in prange(signalsProps.shape[0]):
            _gradientDescendRTPeak(mzs, ints, times, peaksCount, signalsProps, j, interScanMaxSimilarSignalDifferencePPM, minWidth, maxWidth, minRatioFactor, minimumIntensity)

    global _joinSplitPeaks
    @numba.jit(nopython=True, parallel=numbaParallel)
    def _joinSplitPeaks(signalsProps):
        for j in prange(signalsProps.shape[0]):
            for i in range(signalsProps.shape[0]):
                if abs(signalsProps[i, 3] - signalsProps[j, 3]) * 1E6 / signalsProps[j, 3] < signalsProps[j, 5]/2 and \
                    signalsProps[i, 8] <= signalsProps[j,2] and signalsProps[j,2] <= signalsProps[i,11] and \
                    signalsProps[j, 4] < signalsProps[i, 4]:
                    signalsProps[j, 7] = 0
            
    global _integratePeakArea
    @numba.jit(nopython=True, parallel=numbaParallel)
    def _integratePeakArea(mzs, ints, times, peaksCount, peaks):
        for j in prange(peaks.shape[0]):
            peakArea = 0
            scanInd = 0
            for rt in times:
                if peaks[j,2] <= rt <= peaks[j,3]:
                    for mzInd in range(peaksCount[scanInd]):
                        mz = mzs[scanInd, mzInd]
                        if peaks[j,4] <= mz <= peaks[j,5]:
                            peakArea = peakArea + ints[scanInd, mzInd]
                scanInd = scanInd + 1
            peaks[j,6] = peakArea
            
    global _joinSplitPeaksPostProcessing
    @numba.jit(nopython=True, parallel=numbaParallel)
    def _joinSplitPeaksPostProcessing(peaks):
        for j in prange(peaks.shape[0]):
            for i in range(peaks.shape[0]):
                if abs(peaks[i,1] - peaks[j,1]) * 1E6 / peaks[j,1] < (peaks[j,5] - peaks[j,4]) * 1E6 / peaks[j,1] / 2 and \
                    peaks[i, 2] <= peaks[j,0] and peaks[j,0] <= peaks[i,3] and \
                    peaks[j, 6] < peaks[i, 6]:
                    peaks[j, 7] = 0
            
    global _removePeaksWithTooLittleIncrease
    @numba.jit(nopython=True, parallel=numbaParallel)
    def _removePeaksWithTooLittleIncrease(mzs, ints, times, peaksCount, peaks, minimumPeakRatio):
        for j in prange(peaks.shape[0]):
            if peaks[j, 7] > 0:

                # rt, mz, rtstart, rtend, mzstart, mzend, inte
                rtStartInd = _getBestMatch(times, peaks[j, 2])
                rtInd      = _getBestMatch(times, peaks[j, 0])
                mzInd      = _findMostSimilarMZ(mzs, ints, times, peaksCount, rtInd, peaks[j, 4], peaks[j, 5], peaks[j, 1])
                rtEndInd   = _getBestMatch(times, peaks[j, 3])

                ratioOK = 0
                for i in range(rtStartInd, rtEndInd + 1):
                    cmzInd = _findMostSimilarMZ(mzs, ints, times, peaksCount, i, peaks[j, 4], peaks[j, 5], peaks[j, 1])

                    if cmzInd == -1 or ints[i, cmzInd] / ints[rtInd, mzInd] >= minimumPeakRatio:
                        ratioOK = ratioOK + 1

                if ratioOK == 0:
                    peaks[j, 7] = 0
            
    global _renovateArea
    @numba.jit(nopython=True)
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

                    bestSuppInd = _findMostSimilarMZ(mzs, ints, times, peaksCount, rtInd, supMZ - maxOffset, supMZ + maxOffset, supMZ)

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
            
    global _getBestMatch
    @numba.jit(nopython=True)
    def _getBestMatch(vals, val):
        bestMatchInd = -1
        bestMatchDifference = 100000
        for i in range(vals.shape[0]):
            if vals[i] >= 0 and (bestMatchInd == -1 or abs(vals[i] - val) <= bestMatchDifference):
                bestMatchInd = i
                bestMatchDifference = abs(vals[i] - val)
        return bestMatchInd
            
    global _renovatePeaksToArea
    @numba.jit(nopython=True, parallel=numbaParallel)
    def _renovatePeaksToArea(mzs, ints, times, peaksCount, signalsProps, newData, areaRTs, areaMZs, gdProps, fromInd, toInd):

        for instanceInd in prange(fromInd, min(signalsProps.shape[0], toInd)):
            areaTimes = np.zeros(shape = rtSlices, dtype=numba.float32)
            supports = np.zeros(shape = mzSlices, dtype=numba.float32)
            if fromInd <= instanceInd < toInd:
                cInstance = instanceInd - fromInd
                ppmdev    = signalsProps[instanceInd, 5]
                refRT     = signalsProps[instanceInd, 2]
                refMZ     = signalsProps[instanceInd, 3]

                for y in range(mzSlices):
                    supports[y] = 0
                    supports[y] = refMZ*(1. + (-3 + 6*y/(mzSlices-1))*ppmdev/1E6)
                    areaMZs[cInstance, y] = supports[y]
                maxVal = _renovateArea(mzs, ints, times, peaksCount, refRT, supports, newData, cInstance, areaTimes, (supports[1]-supports[0])*8)

                if maxVal > 0:
                    for x in range(rtSlices):
                        areaRTs[cInstance, x] = areaTimes[x]
                        for y in range(mzSlices):
                            newData[cInstance, x, y] /= maxVal

                gdProps[cInstance, 0] = refRT
                gdProps[cInstance, 1] = refMZ
                gdProps[cInstance, 2] = _getBestMatch(areaTimes, signalsProps[instanceInd,  8])
                gdProps[cInstance, 3] = _getBestMatch(areaTimes, signalsProps[instanceInd, 11])
                gdProps[cInstance, 4] = _getBestMatch(supports , refMZ * (1 - ppmdev/2/1E6))
                gdProps[cInstance, 5] = _getBestMatch(supports , refMZ * (1 + ppmdev/2/1E6))

    return _EICWindow, _EICWindowSmoothed

@timeit
def preProcessChromatogram(mzxml, fileIdentifier,
                           intraScanMaxAdjacentSignalDifferencePPM, interScanMaxSimilarSignalDifferencePPM, 
                           RTpeakWidth,
                           minIntensity, exportPath, exportLocalMaxima = "peak-like-shape",
                           minApexBorderRatio = 4,
                           gdMZminRatio = 0.1, gdRTminRatio = 0.01,
                           eicWindowPlusMinus = 30, SavitzkyGolayWindowPlusMinus = 3,
                           instancePrefix = None, exportBatchSize = None,
                           numbaParallel = False, 
                           verbose = True, verbosePrefix = "", verboseTabLog = False):
    rtSlices = peakbot.Config.RTSLICES
    mzSlices = peakbot.Config.MZSLICES
    if exportBatchSize is None:
        exportBatchSize = peakbot.Config.BATCHSIZE
    if instancePrefix is None:
        instancePrefix = peakbot.Config.INSTANCEPREFIX

    tic("preprocessing")
    if verbose:
        print(verbosePrefix, "Preprocessing chromatogram with CPU", sep="")
        print(verbosePrefix, "  | Parameters", sep="")
        print(verbosePrefix, "  | .. intraScanMaxAdjacentSignalDifferencePPM: ", intraScanMaxAdjacentSignalDifferencePPM, sep="")
        print(verbosePrefix, "  | .. interScanMaxSimilarSignalDifferencePPM: ", interScanMaxSimilarSignalDifferencePPM, sep="")
        print(verbosePrefix, "  | .. RTpeakWidth: ", str(RTpeakWidth), sep="")
        print(verbosePrefix, "  | .. minApexBorderRatio: ", minApexBorderRatio, sep="")
        print(verbosePrefix, "  | .. minIntensity: ", minIntensity, sep="")
        s = ""
        try:
            import cpuinfo
            s = cpuinfo.get_cpu_info()["brand_raw"]
        except Exception:
            s = "failed to fetch CPU information"
        print(verbosePrefix, "  | .. device: CPU", s, sep="")
        print(verbosePrefix, "  | .. numba parallel execution: %s"%(numbaParallel), sep="")
        print(verbosePrefix, "  |", sep="")

    _EICWindow, _EICWindowSmoothed = initializeCPUFunctions(gdMZminRatio = gdMZminRatio, gdRTminRatio = gdRTminRatio,
                                                            eicWindowPlusMinus = eicWindowPlusMinus,
                                                            SavitzkyGolayWindowPlusMinus = SavitzkyGolayWindowPlusMinus,
                                                            numbaParallel = numbaParallel)

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
    maxima = np.zeros(mzs.shape, dtype=bool)
    _getLocalMaxima(mzs, ints, times, peaksCount, maxima, intraScanMaxAdjacentSignalDifferencePPM, interScanMaxSimilarSignalDifferencePPM, minIntensity)
    if verbose:
        print(verbosePrefix, "  | Found %d local maxima"%(np.sum(maxima)), sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "n locMax", np.sum(maxima))
        TabLog().addData(fileIdentifier, "locMax (sec)", toc())

    tic()
    maximaPropsAll = np.argwhere(maxima > 0)
    ##       0       1       2   3   4          5                6                         7       8             9          10          11
    ## cols: ind-RT, ind-MZ, RT, MZ, intensity, PPMdevMZProfile, ppmDifferenceAdjacentMZs, isPeak, leftRTBorder, infLeftRT, infRightRT, rightRTBorder
    maximaPropsAll = np.column_stack((maximaPropsAll, np.zeros((maximaPropsAll.shape[0], 20), np.float32)))
    maximaPropsAll = maximaPropsAll.astype(np.float32)
    _getPropsOfSignals(mzs, ints, times, peaksCount, maximaPropsAll)
    if verbose:
        print(verbosePrefix, "  | Calculated properties of local maxima", sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "prop locMax (sec)", toc())

    tic()
    _gradientDescendMZProfileSignals(mzs, ints, times, peaksCount, maximaPropsAll, intraScanMaxAdjacentSignalDifferencePPM)
    maximaProps = maximaPropsAll[maximaPropsAll[:,5]>0,:]
    if verbose:
        print(verbosePrefix, "  | Calculated mz peak deviations", sep="")
        print(verbosePrefix, "  | .. there are %d local maxima with an mz profile peak (%.3f%% of all signals)"%(maximaProps.shape[0], 100.*maximaProps.shape[0]/(maxima.shape[0]*maxima.shape[1])), sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "n locMax mzPeak", maximaProps.shape[0])
        TabLog().addData(fileIdentifier, "locMax mzPeak (sec)", toc())

    tic()
    _gradientDescendRTPeaks(mzs, ints, times, peaksCount, maximaProps, interScanMaxSimilarSignalDifferencePPM, RTpeakWidth[0], RTpeakWidth[1], minApexBorderRatio, minIntensity)
    peaks = maximaProps[maximaProps[:,7]>0,:]
    if verbose:
        print(verbosePrefix, "  | Calculated rt peak borders", sep="")
        print(verbosePrefix, "  | .. there are %d local maxima with an mz profile peak and an rt-peak-like shape (%.3f%% of all signals)"%(peaks.shape[0], 100.*maximaProps.shape[0]/(maxima.shape[0]*maxima.shape[1])), sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "n peaks", peaks.shape[0])
        TabLog().addData(fileIdentifier, "peaks (sec)", toc())

    tic()
    _joinSplitPeaks(peaks)
    peaks = peaks[peaks[:,7]>0,:]
    if verbose:
        print(verbosePrefix, "  | Joined noisy local maxima", sep="")
        print(verbosePrefix, "  | .. there are %d local maxima with an mz profile peak and an rt-peak-like shape that are not split (%.3f%% of all signals)"%(peaks.shape[0], 100.*peaks.shape[0]/(maxima.shape[0]*maxima.shape[1])), sep="")
        print(verbosePrefix, "  | .. took %.1f seconds"%toc(), sep="")
        print(verbosePrefix, "  | ", sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "n join peaks", peaks.shape[0])
        TabLog().addData(fileIdentifier, "join peaks (sec)", toc())

    tic()
    newData = np.zeros((exportBatchSize, rtSlices, mzSlices), np.float32)
    areaRTs = np.zeros((exportBatchSize, rtSlices), np.float32)
    areaMZs = np.zeros((exportBatchSize, mzSlices), np.float32)
    gdProps = np.zeros((exportBatchSize, 6), np.float32) ## gdProps: apexRTInd, apexMZInd, rtLeftBorderInd, rtRightBorderInd, mzLowestInd, mzHighestInd
    outVal = 0

    if exportLocalMaxima is not None:

        locMax = None
        if exportLocalMaxima == "peak-like-shape":
            locMax = peaks
        elif exportLocalMaxima == "localMaxima-with-mzProfile":
            locMax = maximaProps
        elif exportLocalMaxima == "all":
            locMax = maximaPropsAll

        if verbose:
            print(verbosePrefix, "  | Exporting %d local maxima to standardized area"%(locMax.shape[0]), sep="")
            print(verbosePrefix, "  | .. %d batch files, %d features per batch each with approximately %5.1f Mb"%(math.ceil(locMax.shape[0]/exportBatchSize), exportBatchSize, newData.nbytes/1E6), sep="")

        if exportPath is not None:
            if verbose:
                print(verbosePrefix, "  | .. directory '%s'"%(exportPath), sep="")

            cur = 0
            with tqdm.tqdm(total = math.ceil(locMax.shape[0]/exportBatchSize), desc="  | .. exporting", disable=not verbose) as t:
                while cur < locMax.shape[0]:
                    newData.fill(0)
                    areaRTs.fill(-10)
                    areaMZs.fill(0)
                    gdProps.fill(0)

                    ## TODO: Export leaves a lot of GPU resources unused since only exportBatchSize local maxima are processed at a time.
                    ##       There is much room for improvement. The implementation is similar to the training-export
                    _renovatePeaksToArea(mzs, ints, times, peaksCount, locMax, newData, areaRTs, areaMZs, gdProps, cur, cur+exportBatchSize)

                    use = areaRTs[:,0] > -10
                    pickle.dump({"LCHRMSArea": newData[use,:,:],
                                 "areaRTs"   : areaRTs[use,:],
                                 "areaMZs"   : areaMZs[use,:],
                                 "gdProps"   : gdProps[use, :]},
                                open(os.path.join(exportPath, "%s%d.pickle"%(instancePrefix, outVal)), "wb"))

                    cur += exportBatchSize
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

    if verbose:
        print(verbosePrefix, "  | Pre-processing with CPU took %.1f seconds"%toc("preprocessing"), sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "preprocessing (sec)", toc("preprocessing"))

    return peaks, maximaProps, maximaPropsAll






@timeit
def postProcess(mzxml, fileIdentifier, peaks,
                minimumPeakRatio = 0,
                blockdim = 32, griddim = 64,
                eicWindowPlusMinus = 30, SavitzkyGolayWindowPlusMinus = 3,
                verbose = True, verbosePrefix = "", verboseTabLog = False):
    rtSlices = peakbot.Config.RTSLICES
    mzSlices = peakbot.Config.MZSLICES

    tic("postprocessing")
    if verbose:
        print(verbosePrefix, "Postprocessing chromatogram with CPU", sep="")
        print(verbosePrefix, "  |", sep="")

    initializeCPUFunctions(rtSlices, mzSlices, eicWindowPlusMinus, SavitzkyGolayWindowPlusMinus)

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

    maxima = np.zeros(mzs.shape, dtype=bool)
    a = np.zeros([len(peaks), len(peaks[0])], dtype=np.float32)
    for pi, p in enumerate(peaks):
        for i in range(len(p)):
            a[pi,i] = p[i]

    _integratePeakArea(mzs, ints, times, peaksCount, a)
    _joinSplitPeaksPostProcessing(a)
    _removePeaksWithTooLittleIncrease(mzs, ints, times, peaksCount, a, minimumPeakRatio)

    peaks=[]
    for ai in range(a.shape[0]):
        if a[ai,7] > 0:
            peaks.append([a[ai,0], a[ai,1], a[ai,2], a[ai,3], a[ai,4], a[ai,5], a[ai,6]])

    if verbose:
        print(verbosePrefix, "  | Post-processing with CPU took %.1f seconds"%toc("postprocessing"), sep="")
    if verboseTabLog:
        TabLog().addData(fileIdentifier, "postprocessing (sec)", toc("postprocessing"))

    return peaks











@numba.jit(nopython=True, parallel=False)
def _KNNFeaturesCPU(features, featuresOri, neighbors, rtMaxDiff, ppmMaxDiff, nearestNeighbors, rtWeight, mzWeight):
    
    for rowi in prange(features.shape[0]):
        rtR = features[rowi, 2]
        mzR = features[rowi, 3]
        
        for tempi in range(nearestNeighbors):
            neighbors[rowi, tempi, 0] = -1
            neighbors[rowi, tempi, 1] = -1
            neighbors[rowi, tempi, 2] = -1
            neighbors[rowi, tempi, 3] = -1
        
        for coli in range(features.shape[0]):
            rtC = features[coli, 2]
            mzC = features[coli, 3]
            
            rtDiff = abs(rtR - rtC)
            mzDiff = abs(mzR - mzC)
            
            if rowi != coli and rtDiff <= rtMaxDiff and mzDiff*1E6/mzR <= ppmMaxDiff:
                dist = math.sqrt((rtDiff*rtWeight)**2 + (mzDiff*mzWeight)**2)
                for testi in range(nearestNeighbors):
                    if neighbors[rowi, testi, 0] == -1 or dist < neighbors[rowi, testi, 1]:
                        for tempi in range(nearestNeighbors-1, testi, -1):
                            neighbors[rowi, tempi, 0] = neighbors[rowi, tempi-1, 0]
                            neighbors[rowi, tempi, 1] = neighbors[rowi, tempi-1, 1]
                            neighbors[rowi, tempi, 2] = neighbors[rowi, tempi-1, 2]
                            neighbors[rowi, tempi, 3] = neighbors[rowi, tempi-1, 3]
                        neighbors[rowi, testi, 0] = coli
                        neighbors[rowi, testi, 1] = dist
                        neighbors[rowi, testi, 2] = features[coli, 2]
                        neighbors[rowi, testi, 3] = features[coli, 3]
                        break

@numba.jit(nopython=True, parallel=False)
def _updateIdenticalsCPU(features, featuresOri, neighbors, rtMaxDiff, ppmMaxDiff, nearestNeighbors, rtWeight, mzWeight):
    
    for rowi in prange(features.shape[0]):
        
        ## update to mean value
        n = 1
        rtsum = features[rowi, 2]
        mzsum = features[rowi, 3]

        for tempi in range(nearestNeighbors):
            if neighbors[rowi, tempi, 0] > -1:
                n = n + 1
                rtsum = rtsum + neighbors[rowi, tempi, 2]
                mzsum = mzsum + neighbors[rowi, tempi, 3]

        features[rowi, 2] = rtsum / n
        features[rowi, 3] = mzsum / n


@timeit 
def KNNalignFeatures(features, featuresOri, rtMaxDiff, ppmMaxDiff, nearestNeighbors, rtWeight = 1, mzWeight = 0.1, blockdim = 128, griddim = 64):
    
    tic("aligning")
    print("Aligning %d features"%(features.shape[0]))
    
    if len(rtMaxDiff) != len(ppmMaxDiff) or len(ppmMaxDiff) != len(nearestNeighbors):
        raise RuntimeError("Error: the parameters rtMaxDiff, ppmMaxDiff and nearestNeighbors must be arrays of the same length")
        
    neighbors = np.zeros((features.shape[0], max(nearestNeighbors), 4), dtype=np.float32)
    
    for i in range(len(nearestNeighbors)):
        cknn = nearestNeighbors[i]
        crtMaxDiff = rtMaxDiff[i]
        cppmMaxDiff = ppmMaxDiff[i]
        
        print("  | .. iteration %d with %d k-nearest neighbors"%(i+1, cknn))
        
        _KNNFeaturesCPU(features, featuresOri, neighbors, crtMaxDiff, cppmMaxDiff, cknn, rtWeight, mzWeight)        
        _updateIdenticalsCPU(features, featuresOri, neighbors, crtMaxDiff, cppmMaxDiff, cknn, rtWeight, mzWeight)
        
    print("  | .. took %.1f seconds"%(toc("aligning")))
    print("")
    
    return features



        
@numba.jit(nopython=True, parallel=False)
def _groupFeaturesCPU(features, featuresOri, blocks, rtMaxDiff, ppmMaxDiff, rtWeight, mzWeight):
    
    for blocki in prange(blocks.shape[0]):
        
        if blocks[blocki,1] - blocks[blocki,0] == 1:
            ## only a single feature is present, Not much to do
            features[blocks[blocki, 0], 9] = blocks[blocki, 0]
            continue
        
        else:
            ## delete current feature membership
            for rowi in range(blocks[blocki, 0], blocks[blocki, 1]):
                features[rowi, 9] = -1
            
            run = True
            while run:
                ## find the two features closest to each other
                bestDist = 1E8
                bestA = -1
                bestB = -1

                for rowa in range(blocks[blocki,0], blocks[blocki,1]):
                    for rowb in range(rowa+1, blocks[blocki,1]):
                        if features[rowa, 9] == -1 or features[rowb, 9] == -1 or features[rowa, 9] != features[rowb, 9]:
                            rtDiff = abs(features[rowa, 2] - features[rowb, 2])
                            mzDiff = abs(features[rowa, 3] - features[rowb, 3])
                            if rtDiff <= rtMaxDiff and mzDiff * 1E6 / features[rowa, 3] <= ppmMaxDiff:
                                dist = math.sqrt((rtDiff*rtWeight)**2 + (mzDiff*mzWeight)**2)
                                if dist < bestDist:
                                    bestDist = dist
                                    bestA = rowa
                                    bestB = rowb

                if bestA == -1 and bestB == -1:
                    ## No close-by features found, separate each remaining feature into a unique group
                    for rowi in range(blocks[blocki, 0], blocks[blocki, 1]):
                        if features[rowi, 9] == -1:
                            features[rowi, 9] = rowi
                    run = False
                
                elif features[bestA, 9] == -1 and features[bestB, 9] == -1:
                    features[bestA, 9] = bestA
                    features[bestB, 9] = bestA
                elif features[bestA, 9] == -1 and features[bestB, 9] != -1:
                    features[bestA, 9] = features[bestB, 9]
                elif features[bestA, 9] != -1 and features[bestB, 9] == -1:
                    features[bestB, 9] = features[bestA, 9]
                else:
                    cur = features[bestA, 9]
                    curReplace = features[bestB, 9]
                    for rowi in range(blocks[blocki, 0], blocks[blocki, 1]):
                        if features[rowi, 9] == curReplace:
                            features[rowi, 9] = cur
                            

@timeit
def groupFeatures(features, featuresOri, rtMaxDiff, ppmMaxDiff, sampleNameMapping, rtWeight = 1, mzWeight = 0.1, blockdim = 128, griddim = 64):
    # features: ['file', 'Num', 'RT', 'MZ', 'RtStart', 'RtEnd', 'MzStart', 'MzEnd', 'PeakArea', 'featureID', 'use']
    
    tic("grouping")
    print("Grouping features")
    sortOrd = np.squeeze(np.array(features[:,3])).argsort()
    features = features[sortOrd, :]
    featuresOri = featuresOri[sortOrd, :]
    
    blocks = np.zeros((1000, 2), dtype=np.int32) - 1
    last = 0
    cur = 0
    
    for rowi in range(1, features.shape[0]):
        if (features[rowi, 3] - features[rowi - 1, 3]) * 1E6 / features[rowi, 3] >= 3 * ppmMaxDiff:
            if cur >= blocks.shape[0]:
                blocks = np.vstack((blocks, np.zeros((1000, 2), dtype=np.int32) - 1))
            blocks[cur, 0] = last
            blocks[cur, 1] = rowi
            last = rowi
            cur = cur + 1
    if cur >= blocks.shape[0]:
        blocks = np.vstack((blocks, np.zeros((1000, 2), dtype=np.int32) - 1))
    blocks[cur, 0] = last
    blocks[cur, 1] = features.shape[0]
    cur = cur + 1
    blocks = blocks[0:cur,:]
    #print(blocks)
    print("  | .. using %d blocks"%(blocks.shape[0]))
    
    _groupFeaturesCPU(features, featuresOri, blocks, rtMaxDiff, ppmMaxDiff, rtWeight, mzWeight)
    
    print("  | .. creating feature-sample-matrix")
    headers = ["meanRT", "meanMZ", "minRT", "maxRT", "minMZ", "maxMZ", "foundPeak"]
    rows = []
    for sampleNum, sampleName in sampleNameMapping.items():
        headers.append(sampleName)
    for fid in tqdm.tqdm(np.unique(np.array(features[:,9]))):
        tempN = features[features[:,9] == fid,:]
        tempO = featuresOri[features[:,9] == fid,:]
        row = [0 for i in range(7 + len(sampleNameMapping))]
        row[0] = np.mean(tempN[:,2])
        row[1] = np.mean(tempN[:,3])
        
        minRT, maxRT, minMZ, maxMZ, foundPeak = 1E8, 0, 1E8, 0, 0
        for sampleNum, sampleName in sampleNameMapping.items():
            if sampleNum in tempN[:,0]:
                row[7 + sampleNum] = tempN[tempN[:,0] == sampleNum,8][0]
                minRT = min(minRT, tempO[tempN[:,0] == sampleNum,2][0,0])
                maxRT = max(maxRT, tempO[tempN[:,0] == sampleNum,2][0,0])
                minMZ = min(minMZ, tempO[tempN[:,0] == sampleNum,3][0,0])
                maxMZ = max(maxMZ, tempO[tempN[:,0] == sampleNum,3][0,0])
                foundPeak = foundPeak + 1
        
        row[2] = minRT
        row[3] = maxRT
        row[4] = minMZ
        row[5] = maxMZ
        row[6] = foundPeak
                
        rows.append(row)
    
    print("  | .. grouped to %d features"%(len(rows)))
    print("  | .. took %.1f seconds"%(toc("grouping")))
    print("")
    
    return headers, rows
    
    
    
    