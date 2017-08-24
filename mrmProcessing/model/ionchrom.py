#!/usr/bin/env python
# -*- coding: utf-8 -*-

import weakref
from copy import deepcopy

class ionchrom():

    def __init__(self,myid,mz1=0.0,mz2=0.0,ionchromtype="MRM",analname="",
                        analtype="analyte",isionchrom=None,iniprocList=list()):
        self.myid=myid
        self.analname=analname
        self.mz1=mz1
        self.mz2=mz2
        self.ionchromtype=ionchromtype
        self.analtype=analtype
        if isionchrom!=None:
            self.isionchrom=weakref.proxy(isionchrom)
        else:
            self.isionchrom=None
        self.iniprocList=deepcopy(iniprocList)

    def asignISTD(self,isionchrom):
        self.isionchrom=weakref.ref(isionchrom)
    def asignAnalyteType(self,analtype):
        if analtype in ("analyte","is"):
            self.analtype=analtype
            return True
        return False
    def asignIonchromtype(self,ionchromtype):
        if ionchromtype in ("MRM","ExtIon"):
            self.ionchromtype=ionchromtype
            return True
        return False

    def extractRawData(self,mzml):
        if self.ionchromtype=="MRM":
            Q1=self.mz1
            Q3=self.mz2
            #do something
        elif self.ionchromtype=="ExtIon":
            mzFrom=self.mz1
            mzTo=self.mz2
            #do something



