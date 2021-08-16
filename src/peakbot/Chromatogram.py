import sys
import base64
import zlib
import struct
import xml.parsers.expat
from xml.dom.minidom import parse
import numpy as np

import pymzml

from .MSScan import MS1Scan, MS2Scan


# reads and holds the information of one MZXML file
class Chromatogram():
    def __init__(self):
        self.msLevel = 0
        self.current_tag = ''
        self.tag_level = 0
        self.MS1_list = []
        self.MS2_list = []
        self.msInstruments = {}


    def convertChromatogramToNumpyObjects(self, verbose=True, verbosePrefix=""):
        peaksTotal = 0
        maxPeaksPerScan = 0
        minMZ, maxMZ = 1E8, 0

        for scan in self.MS1_list:
            peaksTotal += len(scan.mz_list)
            maxPeaksPerScan = max(maxPeaksPerScan, len(scan.mz_list))

        mzs  = np.zeros((len(self.MS1_list), maxPeaksPerScan), np.float32)
        ints = np.zeros((len(self.MS1_list), maxPeaksPerScan), np.float32)
        times = np.zeros((len(self.MS1_list)), np.float32)
        peaksCount = np.zeros((len(self.MS1_list)), np.int32)

        for scani, scan in enumerate(self.MS1_list):
            mzs[scani, 0:scan.mz_list.shape[0]] = scan.mz_list
            ints[scani, 0:scan.mz_list.shape[0]] = scan.intensity_list
            times[scani] = scan.retention_time
            peaksCount[scani] = scan.mz_list.shape[0]

        t = np.diff(mzs, 1)
        t[t<=0] = 1
        minDiff = np.min(t)
        t = mzs
        t[t<=0] = 1E8
        minMZ = np.min(t)
        maxMZ = np.max(t)

        s = np.diff(times)



        if verbose:
            print(verbosePrefix, "  | .. there are %d scans in the file"%(len(self.MS1_list)), sep="")
            print(verbosePrefix, "  | .. there are %d peaks in the file"%(peaksTotal), sep="")
            print(verbosePrefix, "  | .. the smallest and largest retention time difference of scans are %.2f and %.2f seconds"%(np.min(s), np.max(s)), sep="")
            print(verbosePrefix, "  | .. the mean retention time difference of scans is %.2f with a standard deviation of %.5f"%(np.mean(s), np.std(s)), sep="")
            print(verbosePrefix, "  | .. the highest number of peaks in a scan is %d"%(maxPeaksPerScan), sep="")
            print(verbosePrefix, "  | .. the smallest difference between two signals is %g"%(minDiff), sep="")
            print(verbosePrefix, "  | .. the smallest and largest mz values are %g and %g"%(minMZ, maxMZ), sep="")
            print(verbosePrefix, "  | .. generating two numpy arrays with %d x %d elements"%(mzs.shape[0], ints.shape[1]), sep="")

        return mzs, ints, times, peaksCount

    def keepOnlyFilterLine(self, filterLineToKeep):
        temp = []
        for scan in self.MS1_list:
            if scan.filter_line == filterLineToKeep:
                temp.append(scan)
        self.MS1_list = temp

        temp = []
        for scan in self.MS2_list:
            if scan.filter_line == filterLineToKeep:
                temp.append(scan)
        self.MS2_list = temp

    def removeNoise(self, noiseCutoff=0):
        for scan in self.MS1_list:
            temp = np.argwhere(scan.intensity_list >= noiseCutoff)
            scan.mz_list = scan.mz_list[temp][:,0]
            scan.intensity_list = scan.intensity_list[temp][:,0]
            scan.peak_count = len(temp)

        for scan in self.MS2_list:
            temp = np.argwhere(scan.intensity_list >= noiseCutoff)
            scan.mz_list = scan.mz_list[temp][:,0]
            scan.intensity_list = scan.intensity_list[temp][:,0]
            scan.peak_count = len(temp)

    def removeBounds(self, minRT=0, maxRT=1E6, minMZ=0, maxMZ=1E6):
        use = []
        for scan in self.MS1_list:
            if minRT < scan.retention_time < maxRT:
                temp = np.argwhere(np.logical_and(minMZ < scan.mz_list, scan.mz_list < maxMZ))
                scan.mz_list = scan.mz_list[temp][:,0]
                scan.intensity_list = scan.intensity_list[temp][:,0]
                scan.peak_count = len(temp)
                if scan.peak_count > 0:
                    use.append(scan)
        self.MS1_list = use

        use = []
        for scan in self.MS2_list:
            if minRT < scan.retention_time < maxRT:
                temp = np.argwhere(np.logical_and(minMZ < scan.mz_list, scan.mz_list < maxMZ))
                scan.mz_list = scan.mz_list[temp][:,0]
                scan.intensity_list = scan.intensity_list[temp][:,0]
                scan.peak_count = len(temp)
                if scan.peak_count > 0:
                    use.append(scan)
        self.MS2_list = use


    def getMS1ScanCount(self):
        return len(self.MS1_list)

    def getMS1SignalCount(self):
        c = 0
        for scan in self.MS1_list:
            c += len(scan.mz_list)
        return c

    def getAllMS1Intensities(self):
        h = []
        for scan in self.MS1_list:
            h.extend(scan.intensity_list)
        return h

    def getIthMS1Scan(self, index, filterLine):
        i = 0
        for scan in self.MS1_list:
            if scan.filter_line == filterLine:
                if i == index:
                    return scan
                i += 1
        return None

    def getMS1ScanByNum(self, scanNum):
        return self.MS1_list[scanNum]

    def getClosestMS1Scan(self, scanTime, filterLine = None):
        fscan = self.MS1_list[0]
        fscanInd = 0
        for scanInd, scan in enumerate(self.MS1_list):
            if scan.filter_line == filterLine or filterLine==None:
                if abs(scan.retention_time - scanTime) < abs(fscan.retention_time - scanTime):
                    fscan = scan
                    fscanInd = scanInd
        return fscan, fscanInd

    def getScanByID(self, id):
        for scan in self.MS1_list:
            if scan.id == id:
                return scan
        for scan in self.MS2_list:
            if scan.id == id:
                return scan

    def getFilterLines(self, includeMS1=True, includeMS2=False, includePosPolarity=True, includeNegPolarity=True):
        filterLines = set()
        if includeMS1:
            for scan in self.MS1_list:
                if (includePosPolarity and scan.polarity=="+") or (includeNegPolarity and scan.polarity=="-"):
                    filterLines.add(scan.filter_line)
        if includeMS2:
            for scan in self.MS2_list:
                if (includePosPolarity and scan.polarity=="+") or (includeNegPolarity and scan.polarity=="-"):
                    filterLines.add(scan.filter_line)

        return filterLines

    def getFilterLinesExtended(self, includeMS1=True, includeMS2=False, includePosPolarity=True, includeNegPolarity=True):
        filterLines = {}
        if includeMS1:
            for scan in self.MS1_list:
                if (includePosPolarity and scan.polarity=="+") or (includeNegPolarity and scan.polarity=="-"):
                    if scan.filter_line not in filterLines.keys():
                        filterLines[scan.filter_line]={"scanType": "MS1", "polarity": scan.polarity, "targetStartTime": 10000000, "targetEndTime": 0}
                    filterLines[scan.filter_line]["targetStartTime"]=min(filterLines[scan.filter_line]["targetStartTime"], scan.retention_time)
                    filterLines[scan.filter_line]["targetEndTime"]=min(filterLines[scan.filter_line]["targetEndTime"], scan.retention_time)

        if includeMS2:
            for scan in self.MS2_list:
                if (includePosPolarity and scan.polarity=="+") or (includeNegPolarity and scan.polarity=="-"):
                    if scan.filter_line not in filterLines.keys():
                        filterLines[scan.filter_line]={"scanType": "MS2", "polarity": scan.polarity, "targetStartTime": 10000000, "targetEndTime": 0, "preCursorMz": [], "colisionEnergy": 0}
                    filterLines[scan.filter_line]["targetStartTime"]=min(filterLines[scan.filter_line]["targetStartTime"], scan.retention_time)
                    filterLines[scan.filter_line]["targetEndTime"]=max(filterLines[scan.filter_line]["targetEndTime"], scan.retention_time)
                    filterLines[scan.filter_line]["preCursorMz"].append(scan.precursor_mz)
                    filterLines[scan.filter_line]["colisionEnergy"]=scan.colisionEnergy

            for k, v in filterLines.items():
                if v.scanType=="MS2":
                    v.preCursorMz=sum(v.preCursorMz)/len(v.preCursorMz)

        return filterLines

    def getFilterLinesPerPolarity(self, includeMS1=True, includeMS2=False):
        filterLines = {'+':set(), '-':set()}
        if includeMS1:
            for scan in self.MS1_list:
                filterLines[scan.polarity].add(scan.filter_line)
        if includeMS2:
            for scan in self.MS2_list:
                filterLines[scan.polarity].add(scan.filter_line)

        return filterLines

    def getPolarities(self):
        polarities = set()
        for scan in self.MS1_list:
            polarities.add(scan.polarity)
        for scan in self.MS2_list:
            polarities.add(scan.polarity)

        return polarities

    def getTIC(self, filterLine="", useMS2=False):

        msLevelList=self.MS1_list
        if useMS2:
            msLevelList=self.MS2_list

        TIC     = [0 for i in range(len(self.MS2_list))]
        times   = [0 for i in range(len(self.MS2_list))]
        scanIds = [0 for i in range(len(self.MS2_list))]

        for i, scan in enumerate(msLevelList):
            if filterLine == "" or scan.filter_line == filterLine:
                TIC[i]      = scan.total_ion_current
                times[i]    = scan.retention_time
                scanIds [i] = scan.id

        return TIC, times, scanIds

    # returns a specific area (2-dimensionally bound in rt and mz direction) of the LC-HRMS data
    def getArea(self, startScan, endScan, mz, ppm, filterLine="", intThreshold=0):
        scans = []
        times = []
        scanIDs = []

        scannum = 0
        for scan in self.MS1_list:
            if filterLine == "" or scan.filter_line == filterLine:
                if endScan >= scannum >= startScan:
                    bounds = scan.findMZ(mz, ppm)
                    curScan = []
                    if bounds[0] != -1:
                        for cur in range(bounds[0], bounds[1] + 1):
                            intensity = scan.intensity_list[cur]
                            if intensity >= intThreshold:
                                curScan.append((scan.mz_list[cur], intensity))
                    scans.append(curScan)
                    times.append(scan.retention_time)
                    scanIDs.append(scan.id)

                scannum += 1

        return scans, times, scanIDs

    # returns a specific area (2-dimensionally bound in rt and mz direction) of the LC-HRMS data
    def getSpecificArea(self, startTime, endTime, mzmin, mzmax, filterLine="", intThreshold=0):
        scans = []
        times = []
        scanIDs = []

        for scan in self.MS1_list:
            if filterLine == "" or scan.filter_line == filterLine:
                if startTime <= scan.retention_time <= endTime:
                    bounds = scan._findMZGeneric(mzmin, mzmax)
                    curScan = []
                    if bounds[0] != -1:
                        for cur in range(bounds[0], bounds[1] + 1):
                            intensity = scan.intensity_list[cur]
                            if intensity >= intThreshold:
                                curScan.append((scan.mz_list[cur], intensity))
                    scans.append(curScan)
                    times.append(scan.retention_time)
                    scanIDs.append(scan.id)


        return scans, times, scanIDs

    # returns a specific area (2-dimensionally bound in rt and mz direction) of the LC-HRMS data
    def getSpecificAreaOffset(self, rt, scanWidth, mzmin, mzmax, filterLine=None, intThreshold=0):
        scans = []
        times = []
        scanIDs = []

        bestScan = None
        bestScanRTDiff = -1000000

        for scanInd, scan in enumerate(self.MS1_list):
            if filterLine == None or scan.filter_line == filterLine:
                if abs(scan.retention_time - rt) < abs(bestScanRTDiff):
                    bestScanRTDiff = scan.retention_time - rt
                    bestScan = scanInd
                else:
                    break

        used = 0
        startWith = 0
        for scanInd in range(bestScan-1, 0, -1):
            scan = self.MS1_list[scanInd]
            if filterLine == None or scan.filter_line == filterLine:
                if used < scanWidth:
                    used = used + 1
                    startWith = scanInd
                else:
                    break
        used = 0
        endWith = 0
        for scanInd in range(bestScan+1, len(self.MS1_list), 1):
            scan = self.MS1_list[scanInd]
            if filterLine == None or scan.filter_line == filterLine:
                if used < (scanWidth-1):
                    used = used + 1
                    endWith = scanInd
                else:
                    break

        for scanInd in range(startWith, endWith + 1):
            scan = self.MS1_list[scanInd]
            if filterLine == None or scan.filter_line == filterLine:
                bounds = scan._findMZGeneric(mzmin, mzmax)
                curScan = []
                if bounds[0] != -1:
                    curScan.extend([(scan.mz_list[cur], scan.intensity_list[cur]) for cur in range(bounds[0], bounds[1]+1) if scan.intensity_list[cur] >= intThreshold])
                scans.append(curScan)
                times.append(scan.retention_time)
                scanIDs.append(scan.id)

        return scans, times, scanIDs

    # returns an eic. Single MS peaks and such below a certain threshold may be removed
    def getEIC(self, mz, ppm, filterLine="", removeSingles=True, intThreshold=0, useMS1=True, useMS2=False, startTime=0, endTime=1000000):
        eic = []
        times = []
        scanIds = []

        eicAppend = eic.append
        timesAppend = times.append
        scanIdsAppend = scanIds.append

        if useMS1:
            for scan in self.MS1_list:
                if (scan.filter_line == filterLine or filterLine == "") and startTime <= scan.retention_time <= endTime:
                    bounds = scan.findMZ(mz, ppm)
                    if bounds[0] != -1:
                        eicAppend(max(scan.intensity_list[bounds[0]:(bounds[1] + 1)]))
                    else:
                        eicAppend(0)
                    timesAppend(scan.retention_time)
                    scanIdsAppend(scan.id)

        if useMS2:
            for scan in self.MS2_list:
                if (scan.filter_line == filterLine or filterLine == "") and startTime <= scan.retention_time <= endTime:
                    bounds = scan.findMZ(mz, ppm)
                    if bounds[0] != -1:
                        eicAppend(max(scan.intensity_list[bounds[0]:(bounds[1] + 1)]))
                    else:
                        eicAppend(0)
                    timesAppend(scan.retention_time)
                    scanIdsAppend(scan.id)

        for i in range(1, len(eic)):
            if eic[i] < intThreshold:
                eic[i] = 0

        if removeSingles:
            for i in range(1, len(eic) - 1):
                if eic[i - 1] == 0 and eic[i + 1] == 0:
                    eic[i] = 0

        return eic, times, scanIds


    def getSignalCount(self, filterLine="", removeSingles=True, intThreshold=0, useMS1=True, useMS2=False, startTime=0, endTime=1000000):
        totSignals=0
        for scan in self.MS1_list+self.MS2_list:
            if (scan.filter_line == filterLine or filterLine == "") and startTime <= scan.retention_time <= endTime:
                totSignals+=len([i for i in scan.intensity_list if i>=intThreshold])

        return totSignals


    def getMinMaxAvgSignalIntensities(self, filterLine="", removeSingles=True, intThreshold=0, useMS1=True, useMS2=False, startTime=0, endTime=1000000):
        minInt=10000000000
        maxInt=0
        avgsum=0
        avgcount=0
        for scan in self.MS1_list+self.MS2_list:
            if (scan.filter_line == filterLine or filterLine == "") and startTime <= scan.retention_time <= endTime:

                uVals=[i for i in scan.intensity_list if i>=intThreshold]
                if len(uVals)==0:
                    uVals=[0]

                minInt=min(minInt, min(uVals))
                maxInt=max(minInt, max(uVals))
                avgsum+=sum(uVals)
                avgcount+=len(uVals)

        if avgcount==0:
            return 0, 0, 0
        else:
            return minInt, maxInt, avgsum/avgcount

    # converts base64 coded spectrum in an mz and intensity array
    def decode_spectrum(self, line, peaksCount, compression=None):
        if len(line)>0:
            decoded = base64.b64decode(line)

            if compression=="zlib":
                decoded = zlib.decompress(decoded)

            o = np.ndarray((peaksCount, 2), ">f", decoded, 0)  ## mz values: o[:,0], intensities: o[:,1]
            o = o[o[:,1]>0,:] ## remove empty mz values (these are present in profile mode data)
            return o
        return np.zeros((0,2))

    # xml parser - start element handler
    def _start_element(self, name, attrs):
        self.tag_level += 1
        self.current_tag = name
        #print "start tag", name

        if name == 'parentFile' and "fileName" in attrs.keys():
            self.parentFile = attrs['fileName']

        if name == 'msInstrument' and "msInstrumentID" in attrs.keys():
            self.msInstruments[attrs['msInstrumentID']] = {}
            self.lastMSInstrument = self.msInstruments[attrs['msInstrumentID']]

        if name == 'msModel' and len(self.msInstruments) > 0:
            self.lastMSInstrument['msModel'] = attrs['value']

        if name == 'msMassAnalyzer' and len(self.msInstruments) > 0:
            self.lastMSInstrument['msMassAnalyzer'] = attrs['value']

        if name == 'precursorMz':
            self.MS2_list[-1].precursor_intensity = float(attrs['precursorIntensity'])
            self.MS2_list[-1].precursor_charge = 0
            if "precursorCharge" in attrs.keys():
                self.MS2_list[-1].precursor_charge = int(attrs['precursorCharge'])
            if "activationMethod" in attrs.keys():
                self.MS2_list[-1].activationMethod = str(attrs['activationMethod'])

        if name == 'scan':
            self.curScan = self.curScan + 1

            self.msLevel = int(attrs['msLevel'])
            if self.msLevel == 1:
                tmp_ms = MS1Scan()
            elif self.msLevel == 2:
                tmp_ms = MS2Scan()
            else:
                print("What is it?", attrs)
                sys.exit(1)

            tmp_ms.id = int(attrs['num'])

            tmp_ms.peak_count = int(attrs['peaksCount'])
            tmp_ms.peak_count_tag = int(attrs['peaksCount'])
            tmp_ms.retention_time = float(attrs['retentionTime'].strip('PTS'))
            if tmp_ms.peak_count > 0:
                if "lowMz" in attrs.keys():
                    tmp_ms.low_mz = float(attrs['lowMz'])
                if "highMz" in attrs.keys():
                    tmp_ms.high_mz = float(attrs['highMz'])
                if "basePeakMz" in attrs.keys():
                    tmp_ms.base_peak_mz = float(attrs['basePeakMz'])
                if "basePeakIntensity" in attrs.keys():
                    tmp_ms.base_peak_intensity = float(attrs['basePeakIntensity'])
            tmp_ms.total_ion_current = float(attrs['totIonCurrent'])
            tmp_ms.list_size = 0
            tmp_ms.polarity = str(attrs['polarity'])
            tmp_ms.encoded_mz = ''
            tmp_ms.encoded_intensity = ''
            tmp_ms.encodedData = ''
            tmp_ms.mz_list = []
            tmp_ms.intensity_list = []
            tmp_ms.msInstrumentID = ""
            if "msInstrumentID" in attrs.keys():
                tmp_ms.msInstrumentID = attrs['msInstrumentID']
            tmp_ms.filter_line = "N/A"
            if "filterLine" in attrs.keys():
                tmp_ms.filter_line = attrs['filterLine'] + " (pol: %s)" % tmp_ms.polarity
            #elif len(self.msInstruments) == 1:
            #    tmp_ms.filter_line = "%s (MS lvl: %d, pol: %s)" % (
            #        self.msInstruments.values()[0]["msModel"], self.msLevel, tmp_ms.polarity)
            else:
                tmp_ms.filter_line = "%s (MS lvl: %d, pol: %s)" % (
                    self.msInstruments[tmp_ms.msInstrumentID]["msModel"], self.msLevel, tmp_ms.polarity)

            if self.msLevel == 1:
                self.MS1_list.append(tmp_ms)
            elif self.msLevel == 2:
                if "collisionEnergy" in attrs.keys():
                    tmp_ms.collisionEnergy=float(attrs["collisionEnergy"])
                tmp_ms.ms1_id = self.MS1_list[-1].id
                self.MS2_list.append(tmp_ms)

        if name == "peaks":
            if self.msLevel==1:
                curScan=self.MS1_list[-1]
            elif self.msLevel==2:
                curScan=self.MS2_list[-1]

            curScan.compression=None
            if "compressionType" in attrs.keys():
                curScan.compression=str(attrs["compressionType"])

    # xml parser - end element handler
    def _end_element(self, name):
        #print "end tag", name

        if name == 'scan':
            if self.msLevel==1:
                curScan=self.MS1_list[-1]
            elif self.msLevel==2:
                curScan=self.MS2_list[-1]

            s = self.decode_spectrum(curScan.encodedData, curScan.peak_count, compression=curScan.compression)
            s = s[:,s[1,:] > self.intensityCutoff]
            curScan.mz_list=s[:,0]
            curScan.intensity_list=s[:,1]
            curScan.peak_count=s.shape[1]
            curScan.encodedData=None


        if name == "precursorMz":
            self.MS2_list[-1].precursor_mz = float(self.MS2_list[-1].precursor_mz_data)
            self.MS2_list[-1].filter_line="%s (PreMZ: %.2f ColEn: %.1f ActMet: %s)"%(self.MS2_list[-1].filter_line, self.MS2_list[-1].precursor_mz, self.MS2_list[-1].collisionEnergy, self.MS2_list[-1].activationMethod if hasattr(self.MS2_list[-1], "activationMethod") else "-")

        self.tag_level -= 1
        self.current_tag = ''
        self.msLevel == 0

    # xml parser - CDATA handler
    def _char_data(self, data):

        if self.current_tag == 'precursorMz':
            self.MS2_list[-1].precursor_mz_data+=data

        if self.current_tag == 'peaks' and not self.ignorePeaksData:
            if self.msLevel == 1:
                self.MS1_list[-1].encodedData+=data

            elif self.msLevel == 2:
                self.MS2_list[-1].encodedData+=data

    # parses an mzxml file and stores its information in the respective object.
    # If intensityCutoff is set, all MS peaks below this threshold will be discarded.
    # If ignoreCharacterData is set, only the metainformation will be parsed but not the actual MS data
    def parseMZXMLFile(self, filename_xml, intensityCutoff=-1, ignoreCharacterData=False):
        self.intensityCutoff = intensityCutoff
        self.curScan = 1

        expat = xml.parsers.expat.ParserCreate()
        expat.StartElementHandler = self._start_element
        expat.EndElementHandler = self._end_element
        self.ignorePeaksData=ignoreCharacterData
        expat.CharacterDataHandler = self._char_data

        ##expat.Parse(content_listd)
        expat.ParseFile(open(filename_xml, 'rb'))


    def parseMzMLFile(self, filename_xml, intensityCutoff=-1, ignoreCharacterData=False):

        import pymzml

        run=pymzml.run.Reader(filename_xml)

        for specturm in run:

            try:
                msLevel=int(specturm["ms level"])
            except Exception:
                print("Error: What is it?", specturm["id"], type(specturm), specturm)
                continue

            if msLevel == 1:
                tmp_ms = MS1Scan()
            elif msLevel == 2:
                tmp_ms = MS2Scan()
            else:
                print("What is it?", msLevel, specturm["id"])
                sys.exit(1)

            tmp_ms.id = int(specturm["id"])

            tmp_ms.retention_time = specturm["scan time"]*60
            tmp_ms.filter_line = ""

            tmp_ms.total_ion_current = specturm["total ion current"]
            tmp_ms.list_size = 0
            if specturm["positive scan"] is not None:
                tmp_ms.polarity = "+"
            elif specturm["negative scan"] is not None:
                tmp_ms.polarity = "-"
            else:
                raise RuntimeError("No polarity for scan available")

            tmp_ms.mz_list = [p[0] for p in specturm.peaks("raw")]
            tmp_ms.intensity_list = [p[1] for p in specturm.peaks("raw")]
            tmp_ms.peak_count = len(tmp_ms.mz_list)

            if msLevel == 1:
                self.MS1_list.append(tmp_ms)
            elif msLevel == 2:
                tmp_ms.precursor_mz = specturm["precursors"][0]["mz"]
                tmp_ms.precursor_intensity = 0
                tmp_ms.precursor_charge = specturm["precursors"][0]["charge"]
                self.MS2_list.append(tmp_ms)


    def parse_file(self, filename_xml, intensityCutoff=-1, ignoreCharacterData=False):
        if filename_xml.lower().endswith(".mzxml"):
            return self.parseMZXMLFile(filename_xml, intensityCutoff, ignoreCharacterData)
        elif filename_xml.lower().endswith(".mzml"):
            return self.parseMzMLFile(filename_xml, intensityCutoff, ignoreCharacterData)
        else:
            raise RuntimeError("Invalid file type")


    # updates the data of this MS scan
    # WARNING: unstable, horridly implemented method. SUBJECT OF CHANGE
    def updateScan(self, scan, data):
        scanID = int(scan.getAttribute("num"))

        #  tmp_ms.retention_time = float(attrs['retentionTime'].strip('PTS'))
        if scanID in data.keys() and hasattr(data[scanID], "rt"):
            scan.setAttribute("retentionTime", "PT%.3fS"%(data[scanID].rt))

        peaks = None
        for kid in scan.childNodes:
            if kid.nodeName == "peaks":
                peaks = kid
        assert peaks is not None and len(peaks.childNodes) == 1

        if scanID in data.keys() and len(data[scanID].mzs) > 0:
            h = data[scanID]
            assert len(h.mzs) == len(h.ints)
            peaks.childNodes[0].nodeValue = base64.encodestring("".join([struct.pack(">f", h.mzs[i]) + struct.pack(">f", h.ints[i]) for i in range(len(h.mzs))])).strip().replace("\n", "")
            scan.setAttribute("peaksCount", str(len(h.mzs)))
            scan.setAttribute("lowMz", str(min(h.mzs)))
            scan.setAttribute("highMz", str(max(h.mzs)))
            scan.setAttribute("basePeakMz", "0")
            scan.setAttribute("basePeakIntensity", "0")
            scan.setAttribute("totIonCurrent", str(sum(h.ints)))
        else:
            peaks.childNodes[0].nodeValue = ""
            scan.setAttribute("peaksCount", "0")
            scan.setAttribute("lowMz", "0")
            scan.setAttribute("highMz", "0")
            scan.setAttribute("basePeakMz", "0")
            scan.setAttribute("basePeakIntensity", "0")
            scan.setAttribute("totIonCurrent", "0")

    # updates the data of this mzxml file
    # WARNING: unstable, horridly implemented method. SUBJECT OF CHANGE
    def updateData(self, node, data):
        for kid in node.childNodes:
            if kid.nodeName != '#text':
                if kid.nodeName == "scan":
                    self.updateScan(kid, data)
                else:
                    self.updateData(kid, data)

    # updates the data of this mzxml file
    # WARNING: unstable, horridly implemented method. SUBJECT OF CHANGE
    def resetMZData(self, mzxmlfile, toFile, data):
        dom = parse(mzxmlfile)
        self.updateData(dom, data)
        dom.writexml(open(toFile, "wb"))

    def freeMe(self):
        for scan in self.MS2_list:
            scan.freeMe()
        self.MS2_list = None
        for scan in self.MS1_list:
            scan.freeMe()
        self.MS1_List = None
