#!/usr/bin/env python
# -*- coding: utf-8 -*-

import weakref
from dataHolders import xyDataHolder

class rawdata():

    def __init__(self,myid, mrm,sample,anxyDataHolder=xyDataHolder(""),procList=list()):
        self.myid=myid
        self.chromData=anxyDataHolder
        self.mrm=weakref.ref(mrm)
        self.sample=weakref.ref(sample)
        self.procList=procList
        self.resultList=list()







