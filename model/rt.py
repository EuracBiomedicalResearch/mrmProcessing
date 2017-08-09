#!/usr/bin/env python
# -*- coding: utf-8 -*-

import weakref

class rt():

    def __init__(self,myid,mrm,batch,rtFrom=0.0,rtTo=0.0):
        self.myid=myid
        self.rtFrom=rtFrom
        self.rtTo=rtTo
        self.mrm=weakref.ref(mrm)
        self.batch=weakref.ref(batch)





