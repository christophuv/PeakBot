import peakbot.core
from peakbot.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary

import multiprocessing
import multiprocessing.shared_memory

import numpy as np
import scipy.stats

import math
import platform
import os

@timeit
def initializeNUMPYFunctions(rtSlices = None, mzSlices = None, eicWindowPlusMinus = 30, SavitzkyGolayWindowPlusMinus = 5):
    if rtSlices is None:
        rtSlices = peakbot.Config.RTSLICES
    if mzSlices is None:
        mzSlices = peakbot.Config.MZSLICES
    
    _EICWindow = eicWindowPlusMinus * 2 + SavitzkyGolayWindowPlusMinus * 2 + 1
    _EICWindowSmoothed = eicWindowPlusMinus * 2 + 1

    global populateSharedMemory
    def populateSharedMemory(mzs, ints, times, peaksCount, maxima):
        shm_mzs = multiprocessing.shared_memory.SharedMemory(create=True, size=mzs.nbytes)
        temp = np.ndarray(mzs.shape, dtype=mzs.dtype, buffer=shm_mzs.buf)
        temp[:] = mzs[:]
        mzs = temp

        shm_ints = multiprocessing.shared_memory.SharedMemory(create=True, size=ints.nbytes)
        temp = np.ndarray(ints.shape, dtype=ints.dtype, buffer=shm_ints.buf)
        temp[:] = ints[:]
        ints = temp

        shm_times = multiprocessing.shared_memory.SharedMemory(create=True, size=times.nbytes)
        temp = np.ndarray(times.shape, dtype=times.dtype, buffer=shm_times.buf)
        temp[:] = times[:]
        times = temp

        shm_peaksCount = multiprocessing.shared_memory.SharedMemory(create=True, size=peaksCount.nbytes)
        temp = np.ndarray(peaksCount.shape, dtype=peaksCount.dtype, buffer=shm_peaksCount.buf)
        temp[:] = peaksCount[:]
        peaksCount = temp
        
        shm_maxima = multiprocessing.shared_memory.SharedMemory(create=True, size=maxima.nbytes)
        temp = np.ndarray(maxima.shape, dtype=maxima.dtype, buffer=shm_maxima.buf)
        temp[:] = maxima
        maxima = temp

        return shm_mzs, shm_ints, shm_times, shm_peaksCount, shm_maxima, mzs, ints, times, peaksCount, maxima

    global NPfindMostSimilarMZ
    def NPfindMostSimilarMZ(mzs, ints, times, peaksCount, scanind, mzleft, mzright, mzRef):
        pCount=peaksCount[scanind]    
        if pCount == 0:
            return -1

        min = 0 
        max = pCount

        while min <= max:
            cur = int((max + min) / 2)

            mz = mzs[scanind, cur]
            if mzleft <= mz and mz <= mzright:
                ## find locally most intense signal
                curmz = mzs[scanind, cur]
                curint = ints[scanind, cur]
                while cur-1 > 0 and mzs[scanind, cur-1] >= mzleft and abs(mz - mzs[scanind, cur-1]) < abs(mzRef - mzs[scanind, cur]):
                    cur -= 1
                while cur+1 < pCount and mzs[scanind, cur+1] <= mzright and abs(mz - mzs[scanind, cur+1]) < abs(mzRef - mzs[scanind, cur]):
                    cur += 1

                return cur

            if mzs[scanind, cur] > mzright:
                max = cur - 1
            else:
                min = cur + 1

        return -1

    global NPgetLocalMaxima
    def NPgetLocalMaxima(params):
        shmName_mzs       , shmShape_mzs       , shmDType_mzs       , \
        shmName_ints      , shmShape_ints      , shmDType_ints      , \
        shmName_times     , shmShape_times     , shmDType_times     , \
        shmName_peaksCount, shmShape_peaksCount, shmDType_peaksCount, \
        shmName_maxima    , shmShape_maxima    , shmDType_maxima    , \
        intraScanMaxAdjacentSignalDifferencePPM, interScanMaxSimilarSignalDifferencePPM, minIntensity, startInd, offset = params
        
        ## Unfortunately getting the shared memory in a function does not work. Likely https://bugs.python.org/issue39959
        shm_mzs = multiprocessing.shared_memory.SharedMemory(name=shmName_mzs)
        mzs = np.ndarray(shmShape_mzs, shmDType_mzs, buffer=shm_mzs.buf)

        shm_ints = multiprocessing.shared_memory.SharedMemory(name=shmName_ints)
        ints = np.ndarray(shmShape_ints, shmDType_ints, buffer=shm_ints.buf)

        shm_times = multiprocessing.shared_memory.SharedMemory(name=shmName_times)
        times = np.ndarray(shmShape_times, shmDType_times, buffer=shm_times.buf)

        shm_peaksCount = multiprocessing.shared_memory.SharedMemory(name=shmName_peaksCount)
        peaksCount = np.ndarray(shmShape_peaksCount, shmDType_peaksCount, buffer=shm_peaksCount.buf)
        
        shm_maxima = multiprocessing.shared_memory.SharedMemory(name=shmName_maxima)
        maxima = np.ndarray(shmShape_mzs, bool, buffer = shm_maxima.buf)
        
        if startInd is None:
            startInd = 0
        if offset is None:
            offset = 1
            
        for scani in range(startInd, times.shape[0], offset):
            for peaki in range(0, peaksCount[scani]):
                if peaki < peaksCount[scani] and ints[scani, peaki] >= minIntensity:
                    mz = mzs[scani, peaki]
                    mzDiff = mz*intraScanMaxAdjacentSignalDifferencePPM/1E6

                    mzAdjacentLow = mz*(1.-interScanMaxSimilarSignalDifferencePPM/1E6)
                    mzAdjacentUp  = mz*(1.+interScanMaxSimilarSignalDifferencePPM/1E6)

                    ok = 1

                    if ((peaki-1 > 0 and (ints[scani, peaki-1] < ints[scani, peaki] or (mz-mzs[scani, peaki-1]) > mzDiff)) or peaki == 0) and ((peaki+1 < peaksCount[scani] and (ints[scani, peaki+1] < ints[scani, peaki] or (mzs[scani, peaki+1]-mz) > mzDiff)) or peaki == (peaksCount[scani]-1)):

                        okay = 1 
                        if scani-1 >= 0:
                            pInd = NPfindMostSimilarMZ(mzs, ints, times, peaksCount, scani-1, mzAdjacentLow, mzAdjacentUp, mz)
                            if pInd >= 0:
                                if ints[scani-1, pInd] > ints[scani, peaki]:
                                    okay = 0

                                if okay and pInd-1 >= 0 and ints[scani-1, pInd-1] > ints[scani, peaki] and (mzs[scani-1, pInd] - mzs[scani-1, pInd-1]) < mzDiff:
                                    okay = 0

                                if okay and pInd+1 < peaksCount[scani-1] and ints[scani-1, pInd+1] > ints[scani, peaki] and (mzs[scani-1, pInd+1] - mzs[scani-1, pInd]) < mzDiff:
                                    okay = 0

                        if okay and scani+1 < mzs.shape[0]:
                            pInd = NPfindMostSimilarMZ(mzs, ints, times, peaksCount, scani+1, mzAdjacentLow, mzAdjacentUp, mz)
                            if pInd >= 0:
                                if ints[scani+1, pInd] > ints[scani, peaki]:
                                    okay = 0

                                if okay and pInd-1 >= 0 and ints[scani+1, pInd-1] > ints[scani, peaki] and (mzs[scani+1, pInd] - mzs[scani+1, pInd-1]) < mzDiff:
                                    okay = 0

                                if okay and pInd+1 < peaksCount[scani+1] and ints[scani+1, pInd+1] > ints[scani, peaki] and (mzs[scani+1, pInd+1] - mzs[scani+1, pInd]) < mzDiff:
                                    okay = 0

                        if okay:
                            maxima[scani, peaki] = 1

    global NPgetPropsOfSignals
    def NPgetPropsOfSignals(mzs, ints, times, peaksCount, signalProps):
        for j in range(signalProps.shape[0]):
            scanInd = int(signalProps[j,0])
            peakInd = int(signalProps[j,1])
            signalProps[j,2] = times[scanInd]
            signalProps[j,3] = mzs[scanInd, peakInd]
            signalProps[j,4] = ints[scanInd, peakInd] 

    global NPgradientDescendMZProfile
    def NPgradientDescendMZProfile(mzs, ints, times, peaksCount, signalProps, maxmzDiffPPMAdjacentProfileSignal):
        scanInd = int(signalProps[0])
        peakInd = int(signalProps[1])
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
        scaleFactor =  ints[scanInd, peakInd] / scipy.stats.norm.pdf(mzs[scanInd, peakInd], wMeanMZ, wSTDMZ)

        diffs = 0
        diffsPos = 0
        diffsNeg = 0
        for i in range(indLow, indUp+1):
            hTheo = scipy.stats.norm.pdf(mzs[scanInd, i], wMeanMZ, wSTDMZ) * scaleFactor
            temp = hTheo - ints[scanInd, i]
            diffs = diffs + math.pow(temp,2)
            if temp > 0:
                diffsPos = diffsPos + math.pow(temp,2)
            else:
                diffsNeg = diffsNeg + math.pow(temp, 2)

        return wSTDMZ * 1E6 / mz * 6

    global NPadjacentMZSignalsPPMDifference
    def NPadjacentMZSignalsPPMDifference(mzs, ints, times, peaksCount, signalProps, maxmzDiffPPMAdjacentProfileSignal):
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

    global NPgradientDescendMZProfileSignals
    def NPgradientDescendMZProfileSignals(params):
        
        shmName_mzs        , shmShape_mzs        , shmDType_mzs        , \
        shmName_ints       , shmShape_ints       , shmDType_ints       , \
        shmName_times      , shmShape_times      , shmDType_times      , \
        shmName_peaksCount , shmShape_peaksCount , shmDType_peaksCount , \
        shmName_maximaProps, shmShape_maximaProps, shmDType_maximaProps, \
        intraScanMaxAdjacentSignalDifferencePPM, startInd, offset = params
        
        ## Unfortunately getting the shared memory in a function does not work. Likely https://bugs.python.org/issue39959
        shm_mzs = multiprocessing.shared_memory.SharedMemory(name=shmName_mzs)
        mzs = np.ndarray(shmShape_mzs, shmDType_mzs, buffer=shm_mzs.buf)

        shm_ints = multiprocessing.shared_memory.SharedMemory(name=shmName_ints)
        ints = np.ndarray(shmShape_ints, shmDType_ints, buffer=shm_ints.buf)

        shm_times = multiprocessing.shared_memory.SharedMemory(name=shmName_times)
        times = np.ndarray(shmShape_times, shmDType_times, buffer=shm_times.buf)

        shm_peaksCount = multiprocessing.shared_memory.SharedMemory(name=shmName_peaksCount)
        peaksCount = np.ndarray(shmShape_peaksCount, shmDType_peaksCount, buffer=shm_peaksCount.buf)
        
        shm_maximaProps = multiprocessing.shared_memory.SharedMemory(name=shmName_maximaProps)
        maximaProps = np.ndarray(shmShape_maximaProps, shmDType_maximaProps, buffer = shm_maximaProps.buf)
        
        if startInd is None:
            startInd = 0
        if offset is None:
            offset = 1
            
        for j in range(startInd, maximaProps.shape[0], offset):
            bestPPMFit = NPgradientDescendMZProfile(mzs, ints, times, peaksCount, maximaProps[j,:], intraScanMaxAdjacentSignalDifferencePPM)
            maximaProps[j, 5] = bestPPMFit
            
            ppmDifferenceAdjacentScans = NPadjacentMZSignalsPPMDifference(mzs, ints, times, peaksCount, maximaProps[j,:], intraScanMaxAdjacentSignalDifferencePPM)
            maximaProps[j, 6] = ppmDifferenceAdjacentScans      
    
    global NPSavitzkyGolayFilterSmooth
    def NPSavitzkyGolayFilterSmooth(array, smoothed, elements):
        ## array of size n will be smoothed to size elements (n - 2 * SavitzkyGolayWindowPlusMinus)
        ## Filter of window length 2*SavitzkyGolayWindowPlusMinus+1
        ## TODO: Implementation is static, but could be optimized with scipy.savgol_coeffs, which would also make it more flexible in respect to window sizes
        ## r=8; "smoothed[pos] = " + " + ".join(["%f*array[i%+d]"%(c, i-r) for i, c in enumerate(signal.savgol_coeffs(r*2+1, 3, use="dot"))])
        pos = 0
        for i in range(SavitzkyGolayWindowPlusMinus, SavitzkyGolayWindowPlusMinus + elements):
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
    
    global NPgradientDescendRTPeak
    def NPgradientDescendRTPeak(mzs, ints, times, peaksCount, signalProps, interScanMaxSimilarSignalDifferencePPM, minWidth, maxWidth, minRatioFactor, minimumIntensity):
        scanInd = int(signalProps[0])
        peakInd = int(signalProps[1])
        mz = signalProps[3]
        
        eic = np.zeros(_EICWindow, dtype=np.float32)
        eicSmoothed = np.zeros(_EICWindowSmoothed, dtype=np.float32)

        pos = 0
        for i in range(scanInd - eicWindowPlusMinus - SavitzkyGolayWindowPlusMinus, scanInd + eicWindowPlusMinus + SavitzkyGolayWindowPlusMinus):
            if i >= 0 and i < times.shape[0]:
                ind = NPfindMostSimilarMZ(mzs, ints, times, peaksCount, i, mz * (1-interScanMaxSimilarSignalDifferencePPM/1E6), mz * (1+interScanMaxSimilarSignalDifferencePPM/1E6), mz)
                if ind != -1:
                    eic[pos] = ints[i, ind]
                else:
                    eic[pos] = 0
            pos = pos +1

        NPSavitzkyGolayFilterSmooth(eic, eicSmoothed, _EICWindowSmoothed)

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
        leftOffset = 1
        run = 1
        while run and leftOffset < eicWindowPlusMinus and (a - leftOffset - 1) >= 0 and (e - leftOffset - 1) >= 0:
            der = eicSmoothed[a - leftOffset - 1] / eicSmoothed[a - leftOffset]
            if der > prevDer and eic[e - leftOffset - 1] > 0:
                leftOffset = leftOffset + 1
                prevDer = der
            else:
                run = 0
        inflLeftOffset = leftOffset 
        infDer = prevDer
        run = 1
        while run and leftOffset < eicWindowPlusMinus and (a - leftOffset - 1) >= 0 and (e - leftOffset - 1) >= 0:
            change = eicSmoothed[a - leftOffset] - eicSmoothed[a - leftOffset - 1]
            if change > 0 and eic[e - leftOffset - 1] > 0:
                leftOffset = leftOffset + 1
            else:
                run = 0

        ## find right infliction and border scans
        prevDer = 0
        rightOffset = 1
        run = 1
        while run and rightOffset < eicWindowPlusMinus and (a + rightOffset + 1) < _EICWindowSmoothed and (e + rightOffset + 1) <= _EICWindow:
            der = eicSmoothed[a + rightOffset + 1] / eicSmoothed[a + rightOffset]
            if der > prevDer and eic[e + rightOffset + 1] > 0:
                rightOffset = rightOffset + 1
                prevDer = der
            else:
                run = 0
        inflRightOffset = rightOffset 
        infDer = prevDer
        run = 1
        while run and rightOffset < eicWindowPlusMinus and (a + rightOffset + 1) < _EICWindowSmoothed and (e + rightOffset + 1) <= _EICWindow:
            change = eicSmoothed[a + rightOffset] - eicSmoothed[a + rightOffset + 1]
            if change > 0 and eic[e + rightOffset + 1] > 0:
                rightOffset = rightOffset + 1
            else:
                run = 0

        ## set infliction and border scans as retention times
        signalProps[ 8] = times[min(max(scanInd - leftOffset     , 0), times.shape[0]-1)]
        signalProps[ 9] = times[min(max(scanInd - inflLeftOffset , 0), times.shape[0]-1)]
        signalProps[10] = times[min(max(scanInd + inflRightOffset, 0), times.shape[0]-1)]
        signalProps[11] = times[min(max(scanInd + rightOffset    , 0), times.shape[0]-1)]

        ## calculate and test peak-width and height-ratio (apex to borders)
        peakWidth = rightOffset + leftOffset + 1
        ratio = 0
        infRatioCount = 0
        testInd = scanInd - round(leftOffset * 1.5)
        while testInd < scanInd:
            if testInd >= 0:
                ind = NPfindMostSimilarMZ(mzs, ints, times, peaksCount, testInd, mz * (1-interScanMaxSimilarSignalDifferencePPM/1E6), mz * (1+interScanMaxSimilarSignalDifferencePPM/1E6), mz)
                if ind != -1:
                    ratio = max(ratio, ints[scanInd, peakInd]/ints[testInd, ind])
                else:
                    infRatioCount += 1
            testInd = testInd + 1
        
        testInd = scanInd + round(rightOffset * 1.5)
        while testInd > scanInd:
            if testInd < times.shape[0]:
                ind = NPfindMostSimilarMZ(mzs, ints, times, peaksCount, testInd, mz * (1-interScanMaxSimilarSignalDifferencePPM/1E6), mz * (1+interScanMaxSimilarSignalDifferencePPM/1E6), mz)
                if ind != -1:
                    ratio = max(ratio, ints[scanInd, peakInd]/ints[testInd, ind])
                else:
                    infRatioCount += 1
            testInd = testInd - 1
            
        signalProps[7] = minWidth <= peakWidth and peakWidth <= maxWidth and ints[scanInd, peakInd] >= minimumIntensity and (ratio > minRatioFactor or infRatioCount>3)
    
    global NPgradientDescendRTPeaks 
    def NPgradientDescendRTPeaks(params):
        
        shmName_mzs        , shmShape_mzs        , shmDType_mzs        , \
        shmName_ints       , shmShape_ints       , shmDType_ints       , \
        shmName_times      , shmShape_times      , shmDType_times      , \
        shmName_peaksCount , shmShape_peaksCount , shmDType_peaksCount , \
        shmName_maximaProps, shmShape_maximaProps, shmDType_maximaProps, \
        interScanMaxSimilarSignalDifferencePPM, minWidth, maxWidth, minRatioFactor, minimumIntensity, startInd, offset = params
        
        ## Unfortunately getting the shared memory in a function does not work. Likely https://bugs.python.org/issue39959
        shm_mzs = multiprocessing.shared_memory.SharedMemory(name=shmName_mzs)
        mzs = np.ndarray(shmShape_mzs, shmDType_mzs, buffer=shm_mzs.buf)

        shm_ints = multiprocessing.shared_memory.SharedMemory(name=shmName_ints)
        ints = np.ndarray(shmShape_ints, shmDType_ints, buffer=shm_ints.buf)

        shm_times = multiprocessing.shared_memory.SharedMemory(name=shmName_times)
        times = np.ndarray(shmShape_times, shmDType_times, buffer=shm_times.buf)

        shm_peaksCount = multiprocessing.shared_memory.SharedMemory(name=shmName_peaksCount)
        peaksCount = np.ndarray(shmShape_peaksCount, shmDType_peaksCount, buffer=shm_peaksCount.buf)
        
        shm_maximaProps = multiprocessing.shared_memory.SharedMemory(name=shmName_maximaProps)
        maximaProps = np.ndarray(shmShape_maximaProps, shmDType_maximaProps, buffer = shm_maximaProps.buf)
        
        if startInd is None:
            startInd = 0
        if offset is None:
            offset = 1
            
        for j in range(startInd, maximaProps.shape[0], offset):
            NPgradientDescendRTPeak(mzs, ints, times, peaksCount, maximaProps[j,:], interScanMaxSimilarSignalDifferencePPM, minWidth, maxWidth, minRatioFactor, minimumIntensity)
    
    global NPjoinSplitPeaks
    def NPjoinSplitPeaks(params):
        
        shmName_maximaProps, shmShape_maximaProps, shmDType_maximaProps, \
        startInd, offset = params
        
        shm_maximaProps = multiprocessing.shared_memory.SharedMemory(name=shmName_maximaProps)
        maximaProps = np.ndarray(shmShape_maximaProps, shmDType_maximaProps, buffer = shm_maximaProps.buf)
        
        if startInd is None:
            startInd = 0
        if offset is None:
            offset = 1
            
        for j in range(startInd, maximaProps.shape[0], offset):
            for i in range(maximaProps.shape[0]):
                if abs(maximaProps[i, 3] - maximaProps[j, 3]) * 1E6 / maximaProps[j, 3] < maximaProps[j, 5]/2 and \
                    maximaProps[i, 8] <= maximaProps[j,2] and maximaProps[j,2] <= maximaProps[i,11] and \
                    maximaProps[j, 4] < maximaProps[i, 4]:
                    maximaProps[j,7] = 0
                    
                    

@timeit
def preProcessChromatogram(mzxml, intraScanMaxAdjacentSignalDifferencePPM, interScanMaxSimilarSignalDifferencePPM, RTpeakWidth, 
                           minApexBorderRatio, minIntensity, cpus=1, 
                           rtSlices = None, mzSlices = None, batchSize = None, verbose = True):    
    if rtSlices is None:
        rtSlices = peakbot.Config.RTSLICES
    if mzSlices is None:
        mzSlices = peakbot.Config.MZSLICES
    if batchSize is None:
        batchSize = peakbot.Config.BATCHSIZE
    
    tic("preprocessing")
    if verbose: 
        print("Preprocessing chromatogram with Numpy (optimized, but CPU-based calculations)")
        print("  | Parameters")
        print("  | .. intraScanMaxAdjacentSignalDifferencePPM:", intraScanMaxAdjacentSignalDifferencePPM)
        print("  | .. interScanMaxSimilarSignalDifferencePPM:", interScanMaxSimilarSignalDifferencePPM)
        print("  | .. RTpeakWidth:", str(RTpeakWidth))
        print("  | .. minApexBorderRatio:", minApexBorderRatio)
        print("  | .. minIntensity:", minIntensity)
        print("  | .. cores/threads:", cpus)
        print("  |")
    
    
    initializeNUMPYFunctions()
    
    if verbose:
        print("  | Converted to numpy objects")
    tic()
    mzs, ints, times, peaksCount = mzxml.convertChromatogramToNumpyObjects(verbose = verbose)
    maxima = np.zeros(mzs.shape, dtype=bool)
    if verbose:
        print("  | .. took %.1f seconds"%toc())
        print("  | ")
    

    tic()
    shm_mzs, shm_ints, shm_times, shm_peaksCount, shm_maxima, mzs, ints, times, peaksCount, maxima = peakbot.numpy.populateSharedMemory(mzs, ints, times, peaksCount, maxima)
    if verbose:
        print("  | Copied numpy objects to shared memory")
        print("  | .. took %.1f seconds"%toc())
        print("  | ")

    tic()
    pool = multiprocessing.Pool(cpus)
    pool.imap(NPgetLocalMaxima, 
          [[shm_mzs.name,        mzs.shape,        mzs.dtype,
            shm_ints.name,       ints.shape,       ints.dtype,
            shm_times.name,      times.shape,      times.dtype,
            shm_peaksCount.name, peaksCount.shape, peaksCount.dtype,
            shm_maxima.name,     maxima.shape,     maxima.dtype,
            intraScanMaxAdjacentSignalDifferencePPM, interScanMaxSimilarSignalDifferencePPM, 
            minIntensity, i, cpus] for i in range(cpus)], chunksize=1)
    pool.close()
    pool.join()
    
    maximaProps = np.argwhere(maxima > 0)
    d_maxima = None
    ##       0       1       2   3   4          5                6                         7       8             9          10          11
    ## cols: ind-RT, ind-MZ, RT, MZ, intensity, PPMdevMZProfile, ppmDifferenceAdjacentMZs, isPeak, leftRTBorder, infLeftRT, infRightRT, rightRTBorder
    maximaProps = np.column_stack((maximaProps, np.zeros((maximaProps.shape[0], 20))))  
    shm_maximaProps = multiprocessing.shared_memory.SharedMemory(create=True, size=maximaProps.nbytes)
    temp = np.ndarray(maximaProps.shape, dtype=maximaProps.dtype, buffer=shm_maximaProps.buf)
    temp[:] = maximaProps[:]
    maximaPropsAll = temp
    if verbose:
        print("  | Found %d local maxima"%(maximaPropsAll.shape[0]))
        print("  | .. took %.1f seconds"%toc())
        print("  | ")
    
    tic()
    NPgetPropsOfSignals(mzs, ints, times, peaksCount, maximaPropsAll)
    if verbose:
        print("  | Calculated properties of local maxima")
        print("  | .. took %.1f seconds"%toc())
        print("  | ")
    
    tic()
    pool = multiprocessing.Pool(cpus)
    pool.imap(NPgradientDescendMZProfileSignals, 
          [[shm_mzs.name,         mzs.shape,         mzs.dtype,
            shm_ints.name,        ints.shape,        ints.dtype,
            shm_times.name,       times.shape,       times.dtype,
            shm_peaksCount.name,  peaksCount.shape,  peaksCount.dtype,
            shm_maximaProps.name, maximaProps.shape, maximaProps.dtype,
            intraScanMaxAdjacentSignalDifferencePPM, i, cpus] for i in range(cpus)], chunksize=1)
    pool.close()
    pool.join() 
    
    maximaProps = maximaPropsAll[maximaPropsAll[:,5]>0,:].copy()
    shm_maximaProps.close()
    shm_maximaProps = multiprocessing.shared_memory.SharedMemory(create=True, size=maximaProps.nbytes)
    temp = np.ndarray(maximaProps.shape, dtype=maximaProps.dtype, buffer=shm_maximaProps.buf)
    temp[:] = maximaProps[:]
    maximaProps = temp
    if verbose:
        print("  | Calculated mz peak deviations")
        print("  | .. there are %d local maxima with an mz profile peak (%.3f%% of all signals)"%(maximaProps.shape[0], 100.*maximaProps.shape[0]/(maxima.shape[0]*maxima.shape[1])))
        print("  | .. took %.1f seconds"%toc())
        print("  | ")
    
    
    tic()
    pool = multiprocessing.Pool(cpus)
    pool.imap(NPgradientDescendRTPeaks, 
          [[shm_mzs.name,         mzs.shape,         mzs.dtype,
            shm_ints.name,        ints.shape,        ints.dtype,
            shm_times.name,       times.shape,       times.dtype,
            shm_peaksCount.name,  peaksCount.shape,  peaksCount.dtype,
            shm_maximaProps.name, maximaProps.shape, maximaProps.dtype,
            interScanMaxSimilarSignalDifferencePPM, RTpeakWidth[0], RTpeakWidth[1], minApexBorderRatio, minIntensity,
            i, cpus] for i in range(cpus)], chunksize=1)
    pool.close()
    pool.join() 
    
    maximaProps = maximaProps[maximaProps[:,7]>0,:].copy()
    shm_maximaProps.close()
    shm_maximaProps = multiprocessing.shared_memory.SharedMemory(create=True, size=maximaProps.nbytes)
    temp = np.ndarray(maximaProps.shape, dtype=maximaProps.dtype, buffer=shm_maximaProps.buf)
    temp[:] = maximaProps[:]
    maximaProps = temp
    if verbose:
        print("  | Calculated rt peak borders")
        print("  | .. there are %d local maxima with an mz profile peak and an rt-peak-like shape (%.3f%% of all signals)"%(maximaProps.shape[0], 100.*maximaProps.shape[0]/(maxima.shape[0]*maxima.shape[1])))
        print("  | .. took %.1f seconds"%toc())
        print("  | ")
    
    
    tic()
    pool = multiprocessing.Pool(cpus)
    pool.imap(NPjoinSplitPeaks, 
          [[shm_maximaProps.name, maximaProps.shape, maximaProps.dtype,
            i, cpus] for i in range(cpus)], chunksize=1)
    pool.close()
    pool.join() 
    peaks = maximaProps[maximaProps[:,7]>0,:]
    if verbose:
        print("  | Joined noisy local maxima")
        print("  | .. there are %d local maxima with an mz profile peak and an rt-peak-like shape that are not split (%.3f%% of all signals)"%(peaks.shape[0], 100.*peaks.shape[0]/(maxima.shape[0]*maxima.shape[1])))
        print("  | .. took %.1f seconds"%toc())
        print("  | ")
    
    shm_mzs.close()
    shm_ints.close()
    shm_times.close()
    shm_peaksCount.close()
    shm_maxima.close()
    shm_maximaProps.close()
    
    if verbose: 
        print("  | Processing with numpy took %.1f seconds"%toc("preprocessing"))
        print("")
        
    return peaks, maximaProps, maximaPropsAll






def renovateScan(scan, supports, maxOffset = None):
    if maxOffset is None:
        maxOffset = (supports[1]-supports[0])*8

    newInts = np.zeros((supports.shape[0]))

    mzs = [xt[0] for xt in scan]
    ints = [xt[1] for xt in scan]

    for ind, supmz in enumerate(supports):

        leftmzInd = -1
        rightmzInd = -1

        a = [i-supmz for i in mzs if i <= supmz]
        if len(a) > 0:
            leftmzInd = np.argmin(np.abs(a))
            if abs(mzs[leftmzInd] - supmz) > maxOffset:
                leftmzInd = -1
        a = [i-supmz for i in mzs if i >= supmz]
        if len(a) > 0:
            rightmzInd = np.argmin(np.abs(a)) + sum(1 for i in mzs if i < supmz)
            if abs(mzs[rightmzInd] - supmz) > maxOffset:
                rightmzInd = -1

        newInt = 0
        if leftmzInd == -1 and rightmzInd == -1:
            newInt = 0
        elif leftmzInd == -1 and rightmzInd != -1:
            newInt = ints[rightmzInd]
        elif leftmzInd != -1 and rightmzInd == -1:
            newInt = ints[leftmzInd]
        elif leftmzInd == rightmzInd:
            newInt = ints[leftmzInd]
        else:
            newInt = ints[leftmzInd] + (ints[rightmzInd]-ints[leftmzInd])/(mzs[rightmzInd]-mzs[leftmzInd])*(supmz-mzs[leftmzInd])

        newInts[ind] = newInt

    return newInts
 
def exportAreaForPutativePeak(params):
    mzxml, peaks, startInd, offset, rtScans, mzSlices, batchSize, filterLine, sharedValueLock, sharedValue, sharedPrintLock, verbose, path = params
    dat = None
    properties = None
    packNum = 0
    curPackInd = 0
    for j in range(startInd, peaks.shape[0], offset):
        rt, mz, ppmdev, rtstart, rtend = peaks[j,2], peaks[j,3], peaks[j,5], peaks[j,8], peaks[j,11]
        scans, times, scanIDs = mzxml.getSpecificAreaOffset(rt, math.floor(rtScans/2), 
                                                            mz*(1.-3*ppmdev/1E6), mz*(1.+3*ppmdev/1E6), 
                                                            filterLine=filterLine, intThreshold=1E4)

        supports = np.arange(mz*(1.-3*ppmdev/1E6), mz*(1.+3*ppmdev/1E6)+0.0001, (mz*(1.+3*ppmdev/1E6) - mz*(1.-3*ppmdev/1E6))/mzSlices)
        supports = supports[0:mzSlices]

        if dat is None:
            dat = np.zeros((batchSize, rtScans, mzSlices), dtype=np.float32)
            properties = []
            curPackInd = 0

        for i, s in enumerate(scans):
            s = renovateScan(s, supports)
            dat[curPackInd, i, :] = s
            
        properties.append({"rt": rt, "rtstart": rtstart, "rtend": rtend, "mz": mz, "ppmdev": ppmdev, "scanTimes": times, "support-mzs": supports})
        curPackInd += 1
            
        if curPackInd == batchSize:
            outVal = None
            with sharedValueLock:
                outVal = sharedValue.value
                sharedValue.value += 1
            pickle.dump([dat, properties], open("%s/sample%d.pickle"%(path, outVal), "wb"))
            if verbose and False:
                with sharedPrintLock:
                    print(" .. exported %d local maxima to file %s"%(dat.shape[0], "%s/sample%d.pickle"%(path, outVal)))
            packNum += 1
            curPackInd = 0
            dat = None
            properties = None
    return dat, properties
@timeit
def exportForPeakBot(mzxml, pathTo, expList, filterLine, rtSlices = None, mzSlices = None, batchSize = None, cpus = 12, verbose = False):
    if rtSlices is None:
        rtSlices = peakbot.Config.RTSLICES
    if mzSlices is None:
        mzSlices = peakbot.Config.MZSLICES
    if batchSize is None:
        batchSize = peakbot.Config.BATCHSIZE
    
    pool = multiprocessing.Pool(cpus)
    manager = multiprocessing.Manager()
    sharedValueLock, sharedPrintLock = manager.Lock(), manager.Lock()
    sharedValue = manager.Value('i', 0)
    datAll = None
    propertiesAll = None
    jobsDone, jobsFailed = 0, 0
    result_iter = pool.imap_unordered(exportAreaForPutativePeak, 
                                      [[mzxml, expList, i, cpus, rtScans, mzSlices, batchSize, filterLine, sharedValueLock, sharedValue, sharedPrintLock, verbose, pathTo] for i in range(cpus)], chunksize=1)

    if verbose: print("Exporting %d local maxima on %d CPUs"%(expList.shape[0], cpus))
    # Iterate through all the results
    packNum = 0
    nProcs = cpus
    try:
        while nProcs > 0:
            try:
                # if no timeout is set, Ctrl-C does weird things.
                dat, properties = result_iter.next()

                if datAll is None:
                    datAll = dat
                    propertiesAll = properties
                else:
                    datAll = np.vstack((datAll, dat))
                    propertiesAll.extend(properties)
                jobsDone += 1

                if datAll is not None and datAll.shape[0] >= batchSize:
                    outVal = None
                    with sharedValueLock:
                        outVal = sharedValue.value
                        sharedValue.value += 1
                    pickle.dump([datAll[0:batchSize, :, :], propertiesAll[0:batchSize]], open("%s/sample%d.pickle"%(pathTo, outVal), "wb"))
                    if verbose and False: 
                        with sharedPrintLock:
                            print(" .. exported %d local maxima to file %s"%(datAll.shape[0], "%s/sample%d.pickle"%(pathTo, outVal)))
                    packNum += 1

                    if datAll.shape[0] == batchSize:
                        datAll = None
                        propertiesAll = None
                    else:
                        datAll = datAll[batchSize:,:,:]
                        propertiesAll = propertiesAll[batchSize:]

            except:
                import traceback
                traceback.print_exc()
                jobsFailed += 1
            nProcs -= 1
    except:
        import traceback
        traceback.print_exc()

    if datAll is not None:
        outVal = None
        with sharedLock:
            outVal = sharedValue.value
            sharedValue.value += 1
        pickle.dump([datAll, propertiesAll], open(os.path.join("%s", "sample%d.pickle"%(pathTo, outVal)), "wb"))
        if verbose and False:
            with sharedPrintLock:
                print(" .. exported %d local maxima to file %s"%(datAll.shape[0], os.path.join("%s", "sample%d.pickle"%(pathTo, outVal))))
        datAll = None
        propertiesAll = None
        packNum += 1
        