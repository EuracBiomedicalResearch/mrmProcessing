#!/usr/bin/env python
# -*- coding: utf-8 -*-

import weakref
from dataHolders import xyDataHolder

class rawdata():

    def __init__(self,myid,ionchrom,sample,anxyDataHolder=xyDataHolder("")):
        self.myid=myid
        self.rawData=anxyDataHolder
        self.ionchrom=weakref.proxy(ionchrom)
        self.sample=weakref.proxy(sample)







