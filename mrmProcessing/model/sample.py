#!/usr/bin/env python
# -*- coding: utf-8 -*-

import weakref
import rawdata

class sample():

    def __init__(self,myid,batch,samplName="",samplID="",sampleType="unknown",dataFile=""):
        self.myid=myid
        self.batch=weakref.proxy(batch)
        self.name=samplName
        self.samplID=samplID
        self.sampleType=sampleType
        self.dataFile=dataFile #should be a path to an mzML data file
        self.rawdataList=self.readRawdata()

    def assignSampleType(self,sampleType):
        if sampleType in ("unknown","calibrant","spiked QC","unspiked QC","blank","double blank"):
            self.sampleType=sampleType
            return True
        return False

    def readRawdata(self):
        #read the data into the rawdataList
        for oneIonChrom in self.batch.project.ionchromList:
            rawXYdata=oneIonChrom.extractRawData(mzML)