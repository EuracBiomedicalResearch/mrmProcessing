#!/usr/bin/env python
# -*- coding: utf-8 -*-

import weakref

class processing():

    def __init__(self,myid,rt,rawdata,procType,params):
        self.myid=myid
        self.procType=procType
        self.params=params
        self.rt=weakref.proxy(rt)
        self.rawdata=weakref.proxy(rawdata)







