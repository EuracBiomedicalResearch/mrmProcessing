#!/usr/bin/env python
# -*- coding: utf-8 -*-

import weakref

class result():

    def __init__(self,myid,rt,rawdata,resultType,resultVal,flags=""):
        self.myid=myid
        self.resultType=resultType
        self.resultVal=resultVal
        self.flags=flags
        self.rt=weakref.ref(rt)
        self.rawdata=weakref.ref(rawdata)







