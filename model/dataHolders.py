#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import namedtuple
from numbers import Number
import numpy as np

xyStruct = namedtuple("xyStruct", ["xpoint","ypoint"])

class xyDataHolder:
    def __init__(self, myid):
        self.myid=myid
        self.xyData=[]

    def addPoint(self, x, y):
        self.xyData.append(xyStruct(x,y))

    def addPoints(self,x,y=None):
        if isinstance(x, list) and isinstance(y,list):
            self.addListPoints(x,y)
        elif isinstance(x, Number) and isinstance(y, Number):
            self.addPoint(x,y)
        elif isinstance(x, dict):
            self.addDictPoints(x)

    def addListPoints(self, x, y):
        if len(x)==len(y):
            for i in range(len(x)):
                self.addPoint(x[i],y[i])
        
    def addDictPoints(self,y):
        if isinstance(y, dict):
            for xp,yp in y.iteritems():
                self.addPoint(xp,yp)

    def sumY(self, fromX=-1, toX=-1):
        if fromX==-1:
            fromX=min(self.xyData, key = lambda t: t.xpoint)
        if toX==-1:
            toX=max(self.xyData, key = lambda t: t.xpoint)
        return sum([int(t.xpoint>=fromX and t.xpoint<=toX)*t.ypoint for t in self.xyData])


    def getCopyXYData(self):
        return deepcopy(self.xyData)

    def getXYList(self):
        x=list()
        y=list()
        for oneP in self.xyData:
            x.append(oneP.xpoint)
            y.append(oneP.ypoint)
        return (x,y)

    def getXYDict(self):
        y=dict()
        for oneP in self.xyData:
            y[oneP.xpoint]=oneP.ypoint
        return y

class massChromatogram(xyDataHolder):
    def __init__(self, myid):
        xyDataHolder.__init__(self,myid)
        self.mChromType=None # e.g. 'TIC','SIM','MRM','SRM','XIC'
        self.ion1=None
        self.ion2=None


class massSpectrum(xyDataHolder):
    def __init__(self, myid):
        xyDataHolder.__init__(self,myid)
        self.time=None










    

    