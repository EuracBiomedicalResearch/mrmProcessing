#!/usr/bin/env python
# -*- coding: utf-8 -*-

import batch
import mrm
import rt
import processing

class project():

    def __init__(self,myid,name="",basePath=""):
        self.myid=myid
        self.name=name
        self.basePath=basePath
        self.batchList=list()
        self.mrmList=list()
        self.iniRtList=list()
        self.iniProcList=list()

    def newBatch(self,batchID,batchName="",dataPath=""):
        self.batchList.append(batch.batch(batchID,batchName,dataPath))
    def newMRM(self,mrmID,Q1=0.0,Q3=0.0,name="",mrmtype="analyte",istd=None):
        self.mrmList.append(mrm.mrm(mrmID,Q1,Q3,name,mrmtype,istd))
    def newIniRt(self,rtID,mrmID,rtFrom=0.0,rtTo=0.0):
        self.iniRtList.append(rt.rt(rtID,mrm,self,rtFrom,rtTo))
    def newIniProc(self,procID,rt,procType,params):
        self.iniProcList.append(processing.processing(procID,rt,self,procType,params))
    def readIniRt(self):
        #assumes rt list file in basePath
        rtFile="initialRT.txt"
        #read it into the iniRtList
    def getIniRt(self):
        return self.iniRtList




