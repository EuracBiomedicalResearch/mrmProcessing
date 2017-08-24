#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import weakref
import glob

from dataHolders import xyDataHolder
import rt
import sample
import rawdata

class batch():

    def __init__(self,myid,project,name="",dataPath="",iniRtList=list()):
        self.myid=myid
        self.project=weakref.proxy(project)
        self.name=name
        self.dataPath=dataPath
        self.rtList=deepcopy(iniRtList)
        self.sampleList=list()
        self.rawdataList=list()

    def newRT(self,rtID,ionchrom,rtFrom=0.0,rtTo=0.0):
        self.rtList.append(rt(rtID,ionchrom,self,rtFrom,rtTo))
    def newSample(self,samID, dataFile, samplName,samplID="",sampleType="unknown"):
        self.sampleList.append(sample(samID,self,dataFile,samplName,samplID,sampleType))
    def newRawdata(self,myid,ionchrom,sample,anxyDataHolder=xyDataHolder("")):
        #note: ionchrom here comes from iniRtList as it already connects rt and mz1/mz2 and the processing parameters
        self.rawdataList.append(myid,ionchrom,sample,anxyDataHolder)
    def readDataPath(self):
        #
        #either read the mzML files and add samples and rawdata directly
        #or add samples and give them their datafile path
        #
        #read the injection list for the sample info and add them from there (more complete)
        #or
        #glob mzML from dataPath, create samples based on filename
        #parse the filenames into samples
        #
        print "todo"



