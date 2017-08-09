#!/usr/bin/env python
# -*- coding: utf-8 -*-

import weakref

class mrm():

    def __init__(self,myid,Q1=0.0,Q3=0.0,name="",
                        mrmtype="analyte",istd=None,iniProcList=list()):
        self.myid=myid
        self.name=name
        self.Q1=Q1
        self.Q3=Q3
        self.mrmtype=mrmtype
        self.istd=weakref.ref(istd)
        self.iniProcList=iniProcList

    def asignISTD(self,istd):
        self.istd=weakref.ref(istd)
    def asignMRMtype(self,mrmtype):
        if mrmtype in ("analyte","is"):
            self.mrmtype=mrmtype
            return True
        return False




