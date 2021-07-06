from math import floor

# class for one MS scan
class MSScan():
    def __init__(self):
        self.id = 0
        self.peak_count = 0
        self.filter_line = ''
        self.retention_time = 0.0
        self.low_mz = 0.0
        self.high_mz = 0.0
        self.polarity = ''
        self.base_peak_mz = 0.0
        self.base_peak_intensity = 0.0
        self.total_ion_current = 0.0
        self.list_size = 0
        self.encoded_mz = ''
        self.encoded_intensity = ''
        self.encodedData = ''
        self.mz_list = []
        self.intensity_list = []
        self.msInstrumentID = ""

    # returns the most abundant ms peak in a given range
    def getMostIntensePeak(self, leftInd, rightInd, minInt=0):
        i = -1
        v = -1
        if leftInd != -1 and rightInd != -1:
            for s in range(leftInd, rightInd + 1):
                cInt = self.intensity_list[s]
                if cInt > minInt and cInt > v:
                    i = s
                    v = cInt
        return i

    # HELPER METHOD: tests (and returns) if a given mz value is present in the current ms scan
    # binary search is used for efficiency
    def _findMZGeneric(self, mzleft, mzright, start=None, stop=None):
        
        if len(self.mz_list) == 0:
            return -1, -1

        min = 0 if start is None else start
        max = len(self.mz_list) - 1 if stop is None else stop

        peakCount=len(self.mz_list)

        while min <= max:
            cur = int((max + min) // 2)

            if mzleft <= self.mz_list[cur] <= mzright:
                leftBound = cur
                while leftBound > 0 and self.mz_list[leftBound - 1] >= mzleft:
                    leftBound -= 1

                rightBound = cur
                while (rightBound + 1) < peakCount and self.mz_list[rightBound + 1] <= mzright:
                    rightBound += 1

                return leftBound, rightBound

            if self.mz_list[cur] > mzright:
                max = cur - 1
            else:
                min = cur + 1

        return -1, -1

    # tests (and returns) if a given mz value is present within a given error (in ppm) in the current ms scan
    # start and stop are used for iterative purposes and define starting conditions for the following
    # binary search
    def findMZ(self, mz, ppm, start=None, stop=None):
        mzleft = mz * (1. - ppm / 1000000.)
        mzright = mz * (1. + ppm / 1000000.)

        return self._findMZGeneric(mzleft, mzright, start=start, stop=stop)

    def freeMe(self):
        self.intensity_list = []
        self.mz_list = []


class MS1Scan(MSScan):
    def __init__(self):
        pass


class MS2Scan(MSScan):
    def __init__(self):
        self.ms1_id = 0
        self.precursor_mz = 0.0
        self.precursor_mz_data=""
        self.precursor_intensity = 0.0
        self.precursorCharge = 0
        self.colisionEnergy = 0.
