#!/usr/bin/env python
# -*- coding: utf-8 -*-

import weakref
import rawdata

class sample():

    def __init__(self,myid,batch,dataFile,samplName,samplID="",sampleType="unknown"):
        self.myid=myid
        self.batch=weakref.proxy(batch)
        self.name=samplName
        self.samplID=samplID
        self.sampleType=sampleType
        self.dataFile=dataFile #should be a path to an mzML data file
        self.rawdataList=list()

    def assignSampleType(self,sampleType):
        if sampleType in ("unknown","calibrant","spiked QC","unspiked QC","blank","double blank"):
            self.sampleType=sampleType
            return True
        return False

    def newRawdata(self,myid,ionchrom,sample,anxyDataHolder=xyDataHolder("")):
        #note: somehow we need to be able to connect rawdata, rt and mz1/mz2 and processing parameters for the sample
        #note: ionchrom here should from batch.rtList as it already connects rt and mz1/mz2 and the processing parameters
        #it should also be easy to link those other parameters
        self.rawdataList.append(myid,ionchrom,anxyDataHolder)
    def readRawdata(self):
        #read the data into the rawdataList
        print "todo"
        #for oneIonChrom in self.batch.project.ionchromList:
        #    rawXYdata=oneIonChrom.extractRawData(mzML)