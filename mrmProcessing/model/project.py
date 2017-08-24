#!/usr/bin/env python
# -*- coding: utf-8 -*-

import batch
import ionchrom
import rt
import processing

class project():

    def __init__(self,myid,name="",basePath=""):
        self.myid=myid
        self.name=name
        self.basePath=basePath
        self.batchList=list()
        self.ionchromList=list()
        self.iniRtList=list()
        self.iniProcList=list()

    def newBatch(self,batchID,batchName="",dataPath="",iniRtList=list()):
        self.batchList.append(batch.batch(batchID,self,batchName,dataPath,initRtList))
    def newIonchrom(self,ioncID,mz1=0.0,mz2=0.0,ioncType="MRM",analname="",
                analtype="analyte",isionchrom=None,iniprocList=list()):
        self.ionchromList.append(ionchrom.ionchrom(ioncID,mz1,mz2,ioncType,analname,analtype,isionchrom,iniprocList))
    def newIniRt(self,rtID,ionchrom,rtFrom=0.0,rtTo=0.0):
        self.iniRtList.append(rt.rt(rtID,ionchrom,None,rtFrom,rtTo))
    def newIniProc(self,procID,rt,procType,params):
        self.iniProcList.append(processing.processing(procID,rt,self,procType,params))
    def readIniRt(self):
        #assumes rt list file in basePath
        rtFile="initialRT.txt"
        #read it into the iniRtList
    def getIniRt(self):
        return self.iniRtList




