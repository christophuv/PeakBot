

from collections import OrderedDict
import csv
import time
import datetime
import os

## General functions
#Function statistics and runtime
_functionStats = OrderedDict()
def addFunctionRuntime(fName, duration, invokes = 1):
    if fName not in _functionStats:
        _functionStats[fName]=[0,0]
    _functionStats[fName][0] = _functionStats[fName][0] + duration
    _functionStats[fName][1] = _functionStats[fName][1] + invokes

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        addFunctionRuntime(method.__name__, te-ts, 1)
        return result
    return timed

def printRunTimesSummary():
    print("Recorded runtimes")
    print("-----------------")
    for fName in _functionStats.keys():
        print("%110s: %20s (%10.1f seconds), %9d invokes"%(fName, str(datetime.timedelta(seconds=_functionStats[fName][0])), _functionStats[fName][0], _functionStats[fName][1]))





#Measure runtime of code
__start = {}
def tic(label = "NA"):
    __start[label] = time.time()
def toc(label = "NA"):
    if label not in __start.keys():
        return -1
    return time.time() - __start[label]
def tocP(taskName = "", label="NA"):
    print(" .. task '%s' took %s"%(taskName, str(datetime.timedelta(seconds=toc(label=label)))))
def tocAddStat(taskName = "", label="NA"):
    addFunctionRuntime(taskName, toc(label), 1)






class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TabLog(metaclass = Singleton):

    def __init__(self):
        super(TabLog, self).__init__()

        self.data = {}
        self.instanceOrder = []
        self.keyOrder = []

    def addData(self, instance, key, value, addNumToExistingKeys = True):
        if instance not in self.data.keys():
            self.data[instance] = {}
            self.instanceOrder.append(instance)
        if key in self.data[instance]:
            if not addNumToExistingKeys:
                raise RuntimeError("TabLog: '%s'/'%s' already exists"%(instance, key))
            tkey = "%s (#2)"%(key)
            ih = 2
            while tkey in self.data[instance]:
                ih += 1
                tkey = "%s (#%d)"%(key, ih)
            key = tkey
        if key not in self.keyOrder:
            self.keyOrder.append(key)
        self.data[instance][key] = value
    
    def addSeparator(self):
        self.instanceOrder.append("-!$& separator")

    def print(self):
        widths = {}
        leI = len("Instance")
        for i in self.instanceOrder:
            leI = max(leI, len(str(i)))

        for k in self.keyOrder:
            le = len(str(k))
            for i in self.instanceOrder:
                if i in self.data.keys() and k in self.data[i].keys():
                    le = max(le, len(str(self.data[i][k])))
            widths[k] = le
        
        print("%s  "%(" "*len(str(len(self.instanceOrder)))), end="")
        print("%%-%ds |"%leI%"Instance", end="")
        for k in self.keyOrder:
            print(" %%%ds |"%widths[k]%k, end="")
        print("")

        print("%s--"%("-"*len(str(len(self.instanceOrder)))), end="")
        print("%s-+"%("-"*leI), end="")
        for k in self.keyOrder:
            print("-%s-+"%("-"*widths[k]), end="")
        print("")

        for ind, i in enumerate(self.instanceOrder):
            if i == "-!$& separator":
                print("%s--"%("-"*len(str(len(self.instanceOrder)))), end="")
                print("%s-+"%("-"*leI), end="")
                for k in self.keyOrder:
                    print("-%s-+"%("-"*widths[k]), end="")
            else:
                print("%%%ds. "%len(str(len(self.instanceOrder)))%ind, end="")
                print("%%-%ds | "%leI%i, end="")
                for k in self.keyOrder:
                    if i in self.data.keys() and k in self.data[i].keys():
                        print("%%%ds | "%widths[k]%self.data[i][k], end="")
                    else:
                        print("%%%ds | "%widths[k]%"", end="")
            print("")

    def reset(self):
        self.data = {}
        self.instanceOrder = []
        self.keyOrder = []

    def exportToFile(self, toFile):
        with open(toFile, "w") as fOut:
            tW = csv.writer(fOut, delimiter="\t")
            tW.writerow(["Instance"]+self.keyOrder)
            for ins in self.instanceOrder:
                tW.writerow([ins]+[self.data[ins][key] if key in self.data[ins].keys() else "" for key in self.keyOrder])





def readTSVFile(file, header = True, delimiter = "\t", commentChar = "#", getRowsAsDicts = False, convertToMinIfPossible = False):
    rows = []
    headers = []
    colTypes = {}
    comments = []

    with open(file, "r") as fIn:
        rd = csv.reader(fIn, delimiter=delimiter)

        for rowi, row in enumerate(rd):
            if rowi == 0:
                headers = row

                for celli, cell in enumerate(row):
                    colTypes[celli] = None

            elif row[0].startswith(commentChar):
                comments.append(row)

            else:
                rows.append(row)

                for celli, cell in enumerate(row):
                    if convertToMinIfPossible:
                        colType = colTypes[celli]

                        if colType is None or colType == "int":
                            try:
                                a = int(cell)
                                colType = "int"
                            except Exception:
                                colType = "float"

                        if colType == "float":
                            try:
                                a = float(cell)
                                colType = "float"
                            except Exception:
                                colType = "str"

                        colTypes[celli] = colType

    for rowi, row in enumerate(rows):
        for celli, cell in enumerate(row):
            if colTypes[celli] == "int":
                rows[rowi][celli] = int(cell)
            if colTypes[celli] == "float":
                rows[rowi][celli] = float(cell)

    if getRowsAsDicts:
        temp = {}
        for headeri, header in enumerate(headers):
            temp[header] = headeri
        headers = temp

        temp = []
        for row in rows:
            t = {}
            for header, headeri in headers.items():
                t[header] = row[headeri]
            temp.append(t)
        rows = temp

    return headers, rows


def writeTSVFile(file, headers, rows, delimiter = "\t", commentChar = "#"):
    with open(file, "w") as fOut:
        fOut.write(delimiter.join([str(i) for i in headers]))
        fOut.write("\n")
        for row in rows:
            fOut.write(delimiter.join([str(i) for i in row]))
            fOut.write("\n")
    
