import pymzml
import csv

from base64 import b64decode as b64dec
import zlib
from struct import unpack as unpack

from collections import OrderedDict
#import decimal
import subprocess
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
from numpy import convolve
import scipy as sci
from scipy import signal
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from copy import deepcopy
from timeit import default_timer as timer
from operator import itemgetter, attrgetter, methodcaller
import kde
import myMRM
import re

# class myMRMCollection:
#     def __init__(self, myMRM=None):
#         self.elemenstList=list()
#         self.noupdates=False
#         if myMRM is not None:
#             self.mrmAdd(myMRM)

#     def mrmAdd(self, myMRM):
#         self.elemenstList.append(myMRM)
#         #if not self.noupdates:
#         #    continue



class myLCMSrun:

    def __init__(self, rtFilename):
        self.runFilename=''
        #self.myRun=pymzml.run.Reader(runFilename)
        self.allMRMs=list()
        self.name2mrm=dict()
        self.rtDefs=dict()
        self.processParams=dict()
        self.allres=dict()
        self.allratios=dict()
        self.readRTdefs(rtFilename)
        
        
        
    def readRTdefs(self,rtFilename="LC_RT_Biocrates2.csv",addreplace="replace"):
        if addreplace=="replace":
            self.rtDefs=dict()
            self.processParams=dict()
        with open(rtFilename) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for s in reader:
                if s['Q1'] not in self.rtDefs:
                    self.rtDefs[s['Q1']]=dict()
                    self.processParams[s['Q1']]=dict()
                if s['Q3'] not in self.rtDefs[s['Q1']]:
                    self.rtDefs[s['Q1']][s['Q3']]=dict()
                    self.processParams[s['Q1']][s['Q3']]=dict()
                self.rtDefs[s['Q1']][s['Q3']][s['rt_from']]={'rt_from':float(s['rt_from']),'rt_to':float(s['rt_to']),
                        'name':s['name'],'is_IS':bool(int(s['is_IS'])),'uses_IS':s['uses_IS']}
                self.processParams[s['Q1']][s['Q3']][s['rt_from']]={
                    'fitfunc':s['fit_function'],
                    'spanAll':bool(int(s['span_all'])),
                    'smooth':bool(int(s['smooth'])),
                    'smoothparam':[int(s['smooth_datawindow']),int(s['smooth_repeats']),s['smooth_func']]
                    }
                #if s['name']=='Kynurenine':
                #    print 'csv read'
                #    print s
        f.close

    def sanityCheck(self,runFilename):
        findStr='Initiative Mass Spectrometry Ontology" version="'
        checkOk=True
        p=re.compile(r'[1-9]\.[0-9]')
        with open(runFilename) as f:
            allFile=f.read().splitlines()
            newFile=list()
            for oneLine in allFile:
                newLine=oneLine
                beginPos=oneLine.find(findStr)
                if beginPos>-1:
                    beginPos=beginPos+len(findStr)
                    endPos=oneLine.find('"',beginPos+2)
                    #print beginPos,endPos
                    mtTex=oneLine[beginPos:endPos]
                    
                    if p.match(mtTex)==None:
                        checkOk=False
                        break
                
        f.close()
        return checkOk

    def readData(self,runFilename):
        self.allMRMs=list()
        checkObo=self.sanityCheck(runFilename)
        #if there is an obo use it if not force it to somethig that works
        if checkObo:
            myRun=pymzml.run.Reader(runFilename)
        else:
            myRun=pymzml.run.Reader(runFilename,obo_version='4.0.1')
        cnt=0
        for spectrum in myRun:
            if spectrum['id']!='TIC':
                myID=[item.split("=") for item in spectrum['id'].split(" ")]
                #print myID
                prec=float(myID[2][1])
                prod=float(myID[3][1])
                timeList=list()
                intensityList=list()
                for time, i in spectrum.peaks:
                    timeList.append(time)
                    intensityList.append(i)
                    #print i
                timeList=np.array(timeList)
                intensityList=np.array(intensityList)
                anMRM=myMRM.myMRM(prec,prod,timeList,intensityList)
                
                minRT=9999
                minRT2=9999
                minRTname='None'
                minProcParams=[]
                minis_IS=False
                minuses_IS='0'
                
                prec=myID[2][1]
                prod=myID[3][1]
                minRTname=prec+'_'+prod
                if prec in self.rtDefs:
                    if prod in self.rtDefs[prec]:
                        for oneRt in self.rtDefs[prec][prod]:
                            foneRt=self.rtDefs[prec][prod][oneRt]['rt_from']
                            toneRt=self.rtDefs[prec][prod][oneRt]['rt_to']
                            if anMRM.isRTcontained(foneRt,toneRt):
                                if abs(foneRt-anMRM.aveRT())<minRT:
                                    minRT=foneRt
                                    minRT2=toneRt
                                    minRTname=self.rtDefs[prec][prod][oneRt]['name']
                                    minProcParams=self.processParams[prec][prod][oneRt]
                                    minis_IS=bool(self.rtDefs[prec][prod][oneRt]['is_IS'])
                                    minuses_IS=self.rtDefs[prec][prod][oneRt]['uses_IS']
                #     else:
                #         print prod
                # else:
                #     print prec
                #if minRTname=='Kynurenine':
                #    print minProcParams

                if minRT!=9999:
                    #print 'found', minRTname, minRT
                    anMRM.rt=(minRT,minRT2)
                    anMRM.name=minRTname
                    anMRM.processParams=minProcParams
                    anMRM.is_IS=minis_IS
                    anMRM.uses_IS=minuses_IS
                else:
                    anMRM.name=minRTname

                # else:
                #     print prec, prod
                self.allMRMs.append(anMRM)
                self.name2mrm[anMRM.name]=len(self.allMRMs)-1

    def fitdata(self, idx,showPlot=False):
        return self.allMRMs[idx].makeFit(showPlot=showPlot)


    def fitalldata(self,showPlot=False, makeNew=False):
        if len(self.allres)==0 or makeNew:
            self.allres=dict()
            for i in range(len(self.allMRMs)):
                #print 'readMRM',i
                if self.allMRMs[i].name in ['Spermidine']:
                    self.allres[self.allMRMs[i].name]=self.allMRMs[i].makeFit(showPlot=showPlot)
                else:
                    self.allres[self.allMRMs[i].name]=self.allMRMs[i].makeFit(showPlot=showPlot)
        return self.allres

    def getAllProfiles(self):
        retDict=dict()
        for oneMRM in self.allMRMs:
            retDict[oneMRM.name]=(oneMRM.time,oneMRM.intensity)
        return retDict

    def calcArea(self,mean,sd1,sd2=None):
        area=mean*sd1/0.3989
        if sd2!=None:
            area=(area+mean*sd2/0.3989)/2
        return area
    def calcratios(self):
        if len(self.allres)==0:
            return -1
        else:
            self.allratios=dict()
            for oneMRM in self.allMRMs:
                #oneMRM=self.allMRMs[i]
                if not oneMRM.is_IS:
                    #IS_idx=self.name2mrm[oneMRM.uses_IS]
                    res1=self.allres[oneMRM.name]
                    res2=self.allres[oneMRM.uses_IS]
                    try:
                        ISarea=self.calcArea(res2[1][0],res2[1][2])
                        if ISarea<=0:
                            ISarea=1
                        gaussArea=self.calcArea(res1[1][0],res1[1][2])/ISarea
                    except:
                        gaussArea='N/A'
                    try:
                        ISarea=self.calcArea(res2[3][0],res2[3][2],res2[3][3])
                        if ISarea<=0:
                            ISarea=1
                        bigaussArea=self.calcArea(res1[3][0],res1[3][2],res1[3][3])/ISarea
                    except:
                        bigaussArea='N/A'
                    self.allratios[oneMRM.name]={'gauss':gaussArea,'bigauss':bigaussArea}
        return self.allratios