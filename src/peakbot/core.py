
from collections import OrderedDict
import csv
import time
import datetime

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
    print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
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
                            except:
                                colType = "float"

                        if colType == "float":
                            try:
                                a = float(cell)
                                colType = "float"
                            except:
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
            
                