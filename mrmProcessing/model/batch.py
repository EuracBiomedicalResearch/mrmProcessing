#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import weakref

import rt
import sample

class batch():

    def __init__(self,myid,project,name="",dataPath="",projIniRtList=list()):
        self.myid=myid
        self.project=weakref.proxy(project)
        self.name=name
        self.dataPath=dataPath
        self.rtList=deepcopy(projIniRtList)
        self.sampleList=list()

    def newRT(self,rtID,ionchrom,rtFrom=0.0,rtTo=0.0):
        self.rtList.append(rt(rtID,ionchrom,self,rtFrom,rtTo))
    def newSample(self,samID, samplName="",samplID="",sampleType="unknown"):
        self.sampleList.append(sample(samID,self,samplName,samplID,sampleType))

    def readDataPath(self):
        #glob mzML from dataPath
        #either read the mzML files and add samples and rawdata directly
        #or add samples and give them their datafile path
        print "todo"



