#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.interpolate import CubicSpline
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
import numpy as np
import ransac3

class myMRM:
    def __init__(self, mzPrecursor, mzProduct, timeList, intensityList):
        self.mzPrecursor=mzPrecursor
        self.mzProduct=mzProduct
        self.time=timeList
        self.intensity=intensityList
        self.rt=(-1,-1)
        self.name=''
        self.processParams={'fitfunc':'makeGaussFit','spanAll':False,'smooth':False, 'smoothparam':[]}
        
        
        #self.ransac=ransac3.ransac(threshold=0.025)

    def isRTcontained(self, foneRT, toneRT=None):
        if toneRT==None:
            toneRT=foneRT
        if foneRT>=min(self.time) and toneRT<=max(self.time):
            return True
        return False
    def aveRT(self):
        return np.mean(self.time)

    def find_closest(self,x,xFind):
        y=abs(x-xFind)
        return np.argwhere(y==min(y)).flatten()[0]

    def find_closestMax(self, time=[],intensity=[]):
        if len(time)==0:
            time=self.time
        if len(intensity)==0:
            intensity=self.intensity
        idxFrom=self.find_closest(time,self.rt[0])
        idxTo=self.find_closest(time,self.rt[1])+1
        #print time[idxFrom],time[idxTo]
        localmax=np.array(self.local_maximums(intensity[idxFrom:idxTo],isdict=False).keys())+idxFrom
        #print localmax
        myMax=0
        finalIdx=-1
        for onemax in localmax:
            if intensity[onemax]>myMax:
                myMax=intensity[onemax]
                finalIdx=onemax
        #print idxFrom,idxTo
        #finalIdx=np.argwhere(intensity[localmax]==max(intensity[localmax])).flatten()[0]+idxFrom
        return finalIdx

    def find_baselinedrift(self,x,y,rt_index):

        medy=np.median(y)
        meddy=np.median(abs(y-medy))
        listIndex=list()
        for i in range(len(y)):
            if y[i]<medy+0.5*meddy:
                listIndex.append(i)
        newx=x[listIndex]
        newy=y[listIndex]
        # totn=len(x)
        # totleft=2*int(round(rt_index/4.0,0))
        # totright=rt_index+2*int(round((totn-rt_index)/4.0))
        # newx=np.append(x[:totleft],x[totright:])
        # newy=np.append(y[:totleft],y[totright:])
        return self.ransac.doRansac(newx,newy,n_model=4)

    def find_medianbaseline(self,x,y):

        medy=np.median(y)
        meddy=np.median(abs(y-medy))
        listIndex=list()
        for i in range(len(y)):
            if y[i]<medy+0.5*meddy:
                listIndex.append(i)
        newy=y[listIndex]
        medy=np.median(newy)
        meddy=np.median(abs(newy-medy))
        return medy-2*meddy

    def findLeft(self,y,fromIdx,level,maxUp=1):
        lastY=y[fromIdx]
        if lastY>0:
            z=y/y[fromIdx]
            i=fromIdx
            upCnt=0
            #print 'counting from:',fromIdx, level
            for i in range(fromIdx-1,-1,-1):
                #print i
                if z[i]<level:
                    #i+=1
                    break
                elif (z[i]-lastY)/lastY>0.2:
                    upCnt+=1
                if upCnt>maxUp:
                    i+=1
                    break
                if upCnt==0:
                    lastY=z[i]
        else:
            i=fromIdx-1
        return i

    def findRight(self,y,fromIdx,level,maxUp=1):
        lastY=y[fromIdx]
        if lastY>0:
            z=y/y[fromIdx]
            i=fromIdx
            #print 'counting from:',fromIdx, level
            upCnt=0
            for i in range(fromIdx+1,len(y),1):
                #print i
                if z[i]<level:
                    #i-=1
                    break
                elif (z[i]-lastY)/lastY>0.2:
                    upCnt+=1
                if upCnt>maxUp:
                    i-=1
                    break
                if upCnt==0:
                    lastY=z[i]
        else:
            i=fromIdx+1
        return i 

    def makesmooth(self,values,window,method='movingaverage',params=[]):
        #print window, method, params
        if method=='movingaverage':
            return self.movingaverage(values,window)
        elif method=='movingmedian':
            return self.movingmedian(values,window)
        elif method=='movingweightedaverage':
            return self.movingweightedaverage(values,window)
        elif method=='savitskygol':
            #print len(values)
            if len(params)>0:
                return self.savitskygol(values,window,params[0])
            else:
                return self.savitskygol(values,window,2)
    
    def upsampleCS(self,xdata,ydata,numIncreas=2):
        cs = CubicSpline(xdata, ydata)
        newX=[]
        newY=[]
        for i in range(len(xdata)-1):
            x = np.linspace(xdata[i], xdata[i+1], numIncreas*2)
            newX=np.append(newX,x)
            newY=np.append(newY,cs(x))
        return (newX,newY)

    def upsampleLinear(self,xdata,ydata,numIncreas=2):
        x = np.linspace(0, 2*np.pi, 10)
        newX=[]
        for i in range(len(xdata)-1):
            x = np.linspace(xdata[i], xdata[i+1], numIncreas*2)
            newX=np.append(newX,x)
        newY=np.interp(newX,xdata,ydata)
        return (newX,newY)

    #def movingSDEV(self, values, window):

    def find_line_model(self,x,y):
        # find a line model for these points
        A = np.vstack([x, np.ones(len(x))]).T
        return np.linalg.lstsq(A,y)[0]

    def rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def moving_sdev(self,x,sdev_width):
        if sdev_width % 2 == 0:
            print "stdev width has to be an odd number"
            return x

        movDev=np.std(self.rolling_window(x,sdev_width),1)

        # adding the begin and end numbers to have the 
        # size the same as the original array
        tails=np.ones((sdev_width-1)/2)
        movDev=np.append(movDev[0]*tails,movDev)
        movDev=np.append(movDev,movDev[movDev.size-1]*tails)
        return movDev

    def tophat(self,x,window):
        str_el = np.repeat([1], window)
        return ndimage.white_tophat(x, None, str_el)

    def stage1_fujchrom2016(self,x):
        return self.local_minimums(x)

    def stage2_fujchrom2016(self,x,window):
        x1=self.medSNR_elim(x,window)
        x2=self.firstDeriv_elim(x)
        retDict=dict()
        for i in x1:
            retDict[i]=min(x1[i],x2[i])
        return retDict

    def stage3_fujchrom2016(self,x,y,st2wind):
        ydict=dict()
        for i in range(len(x)):
            ydict[x[i]]=y[i]
        st1=self.stage1_fujchrom2016(ydict)
        #st1b=self.local_maximums(ydict)
        #plt.plot(st1.keys(),st1.values(),'*')
        #plt.plot(st1b.keys(),st1b.values(),'+')
        #plt.ylim(min(y),np.median(y)/10)
        #plt.show()

        st2=self.stage2_fujchrom2016(st1,st2wind)
        #plt.plot(st2.keys(),st2.values(),'+')
        #plt.show()
        xf=list()
        yf=list()
        for oneX in sorted(st2):
            xf.append(oneX)
            yf.append(st2[oneX])
        baseline=np.interp(x,xf,yf)
        baseline=np.minimum(baseline,y)
        return baseline

    def local_outliers(self,x):
        xkeys=list()
        x1=list()
        for oneK in sorted(x.keys()):
            xkeys.append(oneK)
            x1.append(x[oneK])
        x1=np.array(x1)
        xdata=self.rolling_window(x1,3)
        minDict=dict()
        for i in range(xdata.shape[0]):
            oneX=xdata[i,:]
            #print oneX
            if oneX[1]==max(oneX):
                minDict[xkeys[i]]=oneX[1]
            elif oneX[1]==min(oneX):
                minDict[xkeys[i]]=oneX[1]
        return minDict

    def local_maximums(self,x,isdict=True):
        if isdict:
            xkeys=list()
            x1=list()
            for oneK in sorted(x.keys()):
                xkeys.append(oneK)
                x1.append(x[oneK])
        else:
            xkeys=range(len(x))
            x1=x
        x1=np.array(x1)
        xdata=self.rolling_window(x1,3)
        minDict=dict()
        for i in range(xdata.shape[0]):
            oneX=xdata[i,:]
            #print oneX
            if oneX[1]==max(oneX):
                minDict[xkeys[i]]=oneX[1]
        return minDict

    def local_minimums(self,x):
        xkeys=list()
        x1=list()
        for oneK in sorted(x.keys()):
            xkeys.append(oneK)
            x1.append(x[oneK])
        x1=np.array(x1)
        xdata=self.rolling_window(x1,3)
        minDict=dict()
        for i in range(xdata.shape[0]):
            oneX=xdata[i,:]
            #print oneX
            if oneX[1]==min(oneX):
                minDict[xkeys[i]]=oneX[1]
        return minDict

    def median_diffarr(self,x):
        if x.size<2:
            if x.size==1:
                return [0]
            else:
                return []
        else:
            dx=x[1:]-x[0:-1]
            dx=np.append(dx[0],dx)
        return abs(dx-np.median(dx))

    def medSNR_elim(self,x,window=30,prevResult=np.Inf, cnt=0):
        #print 'medSNR begin'
        xkeys=list()
        x1=list()
        for oneK in sorted(x.keys()):
            xkeys.append(oneK)
            x1.append(x[oneK])
        x1=np.array(x1)
        medx=np.median(x1)
        #sanity check...
        if len(x)==0:
            return x
        elif len(x)<window:
            #medDiffArr=np.median(x1)*np.ones(x1.size)
            return x
        else:
            medDiffArr=self.median_diffarr(x1)
        sigma=1.483*np.median(medDiffArr)
        if sigma==0:
            sigma=1
        xdata=self.rolling_window(x1,window)

        beginshape=xdata.shape
        addp=(x1.shape[0]-xdata.shape[0])/2
        
        #print x1.shape, xdata.shape,addp
        for n in range(addp):
            xdata=np.append(xdata[0,:],xdata)
            xdata.shape=(beginshape[0]+(n+1),beginshape[1])
        
        for n in range(addp):
            xdata=np.append(xdata,xdata[-1,:])
            xdata.shape=(beginshape[0]+addp+(n+1),beginshape[1])
        retArr=dict()
        if x1.size>2:
            for i in range(1,x1.size-1):
                #print x1[i-1],x1[i],x1[i+1],sigma
                uno=np.abs(x1[i]-np.median(xdata[i]))/sigma
                due=np.abs(x1[i]-x1[i-1])/sigma
                tre=np.abs(x1[i]-x1[i+1])/sigma
                SNRi = max(uno,due,tre)
                if SNRi>2.5:
                    #np.interp(xkeys[i],(xkeys[i-1],xkeys[i+1]),(x1[i-1],x1[i+1]))
                    retArr[xkeys[i]]=np.interp(xkeys[i],(xkeys[i-1],xkeys[i+1]),(x1[i-1],x1[i+1]))
                else:
                    retArr[xkeys[i]]=x1[i]
            retArr[xkeys[0]]=retArr[xkeys[1]]
            retArr[xkeys[-1]]=retArr[xkeys[-2]]
        else:
            
            for i in range(x1.size):
                retArr[xkeys[i]]=x1[i]
        

        result=np.linalg.norm(np.array(retArr.values())-x1)/np.linalg.norm(x1)
        if result==prevResult or cnt>10:
            return retArr
        elif result > 1e-4:
            retArr=self.medSNR_elim(retArr,window, result,cnt+1)
            return retArr
        return x

    def firstDeriv_elim(self,x):
        xkeys=x.keys()
        x1=np.array(x.values())
        medx=np.median(x1)
        medDiffArr=self.median_diffarr(x1)/(x1+1)
        retArr=dict()
        for i in range(x1.size):
            #print medDiffArr[i]
            if medDiffArr[i]>2.5:
                if i==0:
                    retArr[xkeys[i]]=x1[i+1]
                elif i==x1.size-1:
                    retArr[xkeys[i]]=x1[i-1]
                else:
                    retArr[xkeys[i]]=np.interp(xkeys[i],(xkeys[i-1],xkeys[i+1]),(x1[i-1],x1[i+1]))
            else:
                retArr[xkeys[i]]=x1[i]
        return retArr

    def iterbaseline(self,y,peakwidth,sdev_thresh,sdev_width=3,forceEnds=(False,False)):
        y1, (a1,b1)=self.baseline(y,sdev_thresh,sdev_width, forceEnds)
        s1=self.baselineSplit(y1,peakwidth)
        if s1>-1:
            plt.plot(range(y1.size),y1)
            plt.plot(s1,y1[s1],'r+')
            plt.show()
            y2=self.iterbaseline(y[:s1],peakwidth,sdev_thresh,sdev_width,(True,True))
            y3=self.iterbaseline(y[s1:],peakwidth,sdev_thresh,sdev_width,(True,True))
            #print y2.size,y3.size
            y1=np.append(y2,y3)
        return y1
    def baseline(self, y, sdev_thresh, sdev_width=3, forceEnds=(False,False)):
        if sdev_width % 2 == 0:
            print "stdev width has to be an odd number"
            return y

        maxY=np.max(y)
        if maxY<=0:
            maxY=1
        movDev=self.moving_sdev(y,sdev_width)/maxY
        xbase=np.argwhere(movDev<=sdev_thresh).flatten()
        ybase=y[xbase]
        # if self.name in ['Lys','d6-Orn']:
        #     plt.plot(xbase,ybase)
        #     plt.show()
        if xbase.size>0:
            #fit the points to a line
            a,b=self.find_line_model(xbase,ybase)

            xf=[]
            xe=[]
            if forceEnds[0]:
                #force the beginning if it is not included:
                if xbase[0]!=0:
                    xf=[0,xbase[0]]
                    yf=[y[0],xbase[0]*a+b]
                    af,bf=self.find_line_model(xf,yf)
            if forceEnds[1]:
                #force the end if it is not included:
                if xbase[xbase.size-1]!=len(y)-1:
                    xe=[xbase[xbase.size-1],len(y)-1]
                    ye=[xbase[xbase.size-1]*a+b,y[len(y)-1]]
                    ae,be=self.find_line_model(xe,ye)
            xdata=np.arange(0,y.size)
            calcline=np.ones(y.size)
            for i in range(xdata.size):
                oneX=xdata[i]
                ac=a
                bc=b
                if len(xf)>0:
                    if oneX>xf[0] and oneX<=xf[1]:
                        ac=af
                        bc=bf
                if len(xe)>0:
                    if oneX>xe[0] and oneX<=xe[1]:
                        ac=ae
                        bc=be
                calcline[i]=oneX*ac+bc
            #print xdata.shape
            #print calcline.shape
            #plt.plot(xdata,calcline)
            #plt.plot(xdata,y)
            #plt.show()
            #subtract line from original data
            #calcline=np.arange(0,y.size)*a+b
            ybase=y-calcline

            return (ybase,(a,b))
        return  (y,(1,0))

    def baselineSplit(self,y,width):
        pos=np.argwhere(y>=0).flatten()
        #negative here means consecutive positive numbers in y.
        #numbers >0 means consecutive negative numbers found in y
        Ii = pos[1:]-pos[0:(pos.size-1)]-2
        #now find the maximum number of consecutive negatives in y
        maxnulls=max(Ii)
        maxPos=np.argwhere(Ii==maxnulls).flatten()[0]
        #if the number of consecutive negatives in y surpass the
        #width then split on the most negative number, otherwise return unsplit
        #print maxPos
        #print Ii
        #print y[pos[(maxPos-2):(maxPos+2)]]
        #print pos[(maxPos-2):(maxPos+2)]
        #print maxnulls+1,width
        if maxnulls+1>width:
            miny=min(y)
            splitIdx=np.argwhere(y==miny).flatten()[0]
            #print splitIdx,y.size
            if splitIdx!=0 and splitIdx!=y.size:
                return splitIdx
        return -1
            

    def movingaverage (self, values, window):
        cumsum_vec = np.cumsum(np.insert(values, 0, 0)) 
        ma_vec = (cumsum_vec[window:] - cumsum_vec[:-window]) / window
        adds=(window-1)/2
        ma_vec=np.append(values[0:adds],ma_vec)
        ma_vec=np.append(ma_vec,values[(len(values)-adds):])
        return ma_vec

    def movingweightedaverage(self,values,window):
        adds=(window-1)/2
        weights=self.makeWeights(adds,0,window)
        # retList=list()
        # for i in range(adds,len(values)-adds):
        #     fromIdx=i-adds
        #     toIdx=i+adds+1
            
        #     retList.append(np.average(values[fromIdx:toIdx],weights=weights))
        retVec=np.average(self.rolling_window(values,window),1)
        retVec= np.append(values[0:adds],retVec)
        retVec= np.append(retVec,values[(len(values)-adds):])
        return retVec

    def movingmedian(self, values, window):
        return signal.medfilt(values,window)
    def savitskygol(self, values, window, polyorder=2):
        #try:    
        return signal.savgol_filter(values,window,polyorder)
        #except TypeError:


    def fgauss(self, x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2.*sigma**2))

    def bigauss(self,x,a,x0,sigma1,sigma2):
        y=[self.fgauss(onex,a,x0,sigma1) if onex<x0 else self.fgauss(onex,a,x0,sigma2) for onex in x]
        return np.array(y)

    def makeWeights(self,yIdx,fromIdx,toIdx):
        sigmadata=list()
        for i in range(fromIdx,toIdx):
            if i!=yIdx:
                sigmadata.append(1.0/2**abs(i-yIdx))
            else:
                sigmadata.append(1)
        return np.array(sigmadata)

    def makeFit(self, showPlot=False):
        if self.processParams['fitfunc']=='makeGaussFit':
            return self.makeGaussFit(showPlot)

    def makeGaussFit(self, showPlot=False):
        if self.rt!=(-1,-1):
            #print self.name
            #print 'makeGaussFit begin',self.name
            #if showPlot:
                #print self.processParams['smoothparam']
            spanAll=self.processParams['spanAll']
            smooth=self.processParams['smooth']
            smoothparam=self.processParams['smoothparam']

            #yIdx=self.find_closestMax()
            #idxFrom=self.find_closest(self.time,self.rt[0]-0.05)
            #idxTo=self.find_closest(self.time,self.rt[1]+0.05)+1
            #medSignal=np.median(self.intensity[idxFrom:idxTo])+1e-16
            #maxSignal=self.intensity[yIdx]
            #detectPossible=maxSignal/medSignal>3
            

            #yIdx=np.argwhere(self.intensity[xIdx-6:xIdx+7]==max(self.intensity[xIdx-6:xIdx+7])).flatten()[0]
            #yIdx+=xIdx-6
            #print 'data extraction'
            intensity=self.intensity[:]
            time=self.time[:]
            time,intensity=self.upsampleLinear(time,intensity,1.5)
            if smooth:
                for i in range(smoothparam[1]):
                    intensity=self.makesmooth(intensity,smoothparam[0],smoothparam[2])

            intensbefore=intensity[:]
            intensity,(a,b)=self.baseline(intensity,0.0005,5)
            yIdx=self.find_closestMax(time,intensity)
            # if yIdx>-1:
            #     print self.name,'closestMax ok'
            # else:
            #     print self.name, 'closestMax not ok'
            maxSignal=0
            noiseSignal=np.Inf
            foundLocal=False
            fromIdx=0
            toIdx=len(intensity)
            xdata=time[fromIdx:toIdx]
            ydata=intensity[fromIdx:toIdx]
            if yIdx>-1:
                noiseSignal=a*yIdx+b
                if noiseSignal<1:
                    noiseSignal=1
                maxSignal=intensity[yIdx]
                if not spanAll:
                        fromIdx=self.findLeft(intensity,yIdx,0.2)
                        toIdx=self.findRight(intensity,yIdx,0.2)+1
                        if toIdx-fromIdx<5:
                            #print yIdx, fromIdx, toIdx, self.intensity[fromIdx-5:toIdx+5]
                            diff=int(round((6.0-(toIdx-fromIdx))/2,0))
                            fromIdx-=diff
                            toIdx+=diff
                        if fromIdx<0:
                            fromIdx=0
                        if toIdx>len(intensity):
                            toIdx=len(intensity)+1
                else:
                    fromIdx=0
                    toIdx=len(intensity)


                xdata=time[fromIdx:toIdx]
                ydata=intensity[fromIdx:toIdx]

                localmax=self.local_maximums(ydata,isdict=False)
                totn=len(ydata)
                
                for oneMax in localmax:
                    if totn-oneMax>2 and oneMax>1:
                        foundLocal=True

            #print self.name,'S/N',maxSignal, noiseSignal, round(maxSignal/noiseSignal,0)
            #print self.name,'foundLocal', foundLocal
            detectPossible=maxSignal/noiseSignal>3 and foundLocal
            gaussRes=0
            bigaussRes=0
            popt=(0,0,0)
            popt2=(0,0,0,0)
            if detectPossible:
            #orgxdata,orgydata=(orgTime[fromIdx:toIdx],orgInt[fromIdx:toIdx])
            #xdata,ydata=self.upsampleCS(xdata,ydata)
            #xdata,ydata=self.upsampleLinear(xdata,ydata)
            #print 'smoothing'
            #if smooth:
                #ydata=self.intensity[fromIdx:toIdx] 
                #for i in range(smoothparam[1]):
                    #ydata=self.makesmooth(ydata,smoothparam[0],smoothparam[2])
                    #print i
                    #ydata=np.append(self.intensity[fromIdx],ydata)
                    #ydata=np.append(ydata,self.intensity[toIdx-1])
                #xdata=self.time[fromIdx:toIdx]
            #
            #xdata,ydata=self.upsampleLinear(xdata,ydata,1.5)
                mean=time[yIdx]
                a=intensity[yIdx]
            #b=np.median(ydata)
            #baseline=min(ydata)
            #orgydata=orgydata-baseline
            #ydata=ydata-baseline
            #sigma=0.01
            #mean = sum(xdata * ydata)/sum(xdata)
                
                popt=(maxSignal,time[yIdx],0)
                popt2=(maxSignal,time[yIdx],0,0)
            
                sigma = np.std(xdata-mean)/2
                #sigma2=deepcopy(sigma)
                #print 'gauss opt',self.name
                try:
                    popt,pcov = curve_fit(self.fgauss,xdata,ydata,p0=[a,mean,sigma], bounds=[0,np.inf])
                    gaussRes=1
                except RuntimeError:
                    #print "gauss fit failed"
                    gaussRes=-1
                    popt=[a,mean,sigma]
                #print [a,mean,sigma]
                #print popt
                #[popt[0],potp[1],popt[2],sigma2]
                #[a,mean,sigma,sigma2*1.05]
                
                alpha=popt[1]
                sigma1,sigma2=self.calc_sigma12(xdata,ydata,alpha)
                delta=self.find_delta(xdata,ydata,alpha,sigma1,sigma2)
                popt3=[delta,alpha,sigma1,sigma2]
                #yIdx2=np.argwhere(ydata==max(ydata)).flatten()[0]
                #fromIdx2=self.findLeft(ydata,yIdx2,0.2)
                #toIdx2=self.findRight(ydata,yIdx2,0.2)+1
                #print(ydata[fromIdx2]/ydata[yIdx2])
                #popt4=self.find_bigauss(xdata,ydata)
                #var=0.045
                #bounds=([0,mean*(1-var),0,0],[np.inf,mean*(1+var),np.inf,np.inf])
                bounds=(0,np.inf)
                #print 'bigauss opt',self.name
                try:
                    if spanAll:
                        popt2,pcov2 = curve_fit(self.bigauss,xdata,ydata,p0=popt3, #[a,mean,sigma*0.85,sigma*0.85], 
                            bounds=bounds
                            )
                    else:
                        popt2,pcov2 = curve_fit(self.bigauss,xdata,ydata,p0=popt3, #[a,mean,sigma*0.85,sigma*0.85], 
                                    bounds=bounds, 
                                    #method='dogbox'
                                    #sigma=self.makeWeights(yIdx,fromIdx,toIdx), 
                                    #absolute_sigma=False
                            )
                    bigaussRes=1
                except RuntimeError:
                    #print "bigauss fit failed"
                    bigaussRes=-1
                    popt2=popt3 #[a,mean,sigma,sigma,b,0]
            
            #print popt3
            #print popt4
            #print popt2
            #print gaussRes,popt[3:5]
            #print bigaussRes,popt2[4:6]
            # sqrtTerm=np.sqrt(-4*2*np.log(0.25)*optSigma**2)
            # optA25=optRt-(2*optRt-sqrtTerm)/2
            # optB25=(2*optRt+sqrtTerm)/2-optRt
            # A25=optRt-self.time[self.findLeft(self.intensity,xIdx,0.25)]
            # B25=self.time[self.findRight(self.intensity,xIdx,0.25)]-optRt
            # print A25/optA25,B25/optB25
            #print popt
            #popt=[a,mean,sigma]
            #print 'plot'
            
            if showPlot:
                delta=7
                plotfromIdx=self.find_closest(self.time,time[fromIdx])-delta
                if plotfromIdx<0:
                    plotfromIdx=0
                plottoIdx=self.find_closest(self.time,time[toIdx-1])+delta
                if plottoIdx>=len(self.time):
                    plottoIdx=len(self.time)-1
                
                fitxdata=np.arange(xdata[0]-0.1,xdata[len(xdata)-1]+0.1,step=0.001)
                #plt.plot(plotxdata,plotydata,'b+',label='data')
                
                plt.plot(xdata,ydata,'g+',label='processed data')
                plt.plot(self.time[plotfromIdx:plottoIdx],self.intensity[plotfromIdx:plottoIdx],'g',label='Raw data')
                #plt.plot(orgxdata,orgydata,'b+',label='org data')
                #plt.plot(xdata[fromIdx2:toIdx2],ydata[fromIdx2:toIdx2],'g+',label='Bigauss data')
                if detectPossible:
                    plt.plot(fitxdata,self.fgauss(fitxdata,*popt),'r:',label='Gaussian fit')
                    plt.plot(fitxdata,self.bigauss(fitxdata,*popt2),'b:',label='Bi-Gaussian fit')
                    plt.title(self.name+': '+str(self.mzPrecursor)+'-'+str(self.mzProduct))
                else:
                    plt.title(self.name+': '+str(self.mzPrecursor)+'-'+str(self.mzProduct)+'\nS/N<3 or no local maxima')
                    #xdata=self.time[idxFrom:idxTo]
                    #ydata=self.intensity[idxFrom:idxTo]
                    plt.plot(self.time[plotfromIdx:plottoIdx],self.intensity[plotfromIdx:plottoIdx],'r:',label='Raw data')
                #plt.plot(fitxdata,self.bigauss(fitxdata,*popt3),'g:',label='Bi-Gaussian calc')
                plt.legend()
                plt.xlabel('Time (min)')
                plt.ylabel('Ion counts (au)')
                plt.show()

            #print 'done',self.name
            return (gaussRes, popt, bigaussRes, popt2)
        else:
            #print 'no rt found for:'+self.name+'-'+str(self.mzPrecursor)+'-'+str(self.mzProduct)
            return (-2,(-1,0,0),-2,(-1,0,0,0))

    def calc_dti(self,t):
        dti=list()
        n=len(t)
        for i in range(n):
            if i==0:
                dti.append(t[1]-t[0])
            elif i==len(t)-1:
                dti.append((t[(n-1)]-t[(n-2)])/2)
            else:
                dti.append(t[(i+1)]-t[i-1])
        return np.array(dti)

    def calc_aTau(self,xdata,ydata,tau):
        dtidata=self.calc_dti(xdata)
        sum1=1e-16
        sum2=1e-16
        for i in range(len(xdata)):
            if xdata[i]<tau:
                sum1+=ydata[i]*dtidata[i]
            else:
                sum2+=ydata[i]*dtidata[i]
        return np.log(sum1)-np.log(sum2)

    def calc_bTau(self,xdata,ydata,tau):
        dtidata=self.calc_dti(xdata)
        sum1=1e-16
        sum2=1e-16
        for i in range(len(xdata)):
            if xdata[i]<tau:
                sum1+=ydata[i]*dtidata[i]*(xdata[i]-tau)**2
            else:
                sum2+=ydata[i]*dtidata[i]*(xdata[i]-tau)**2+1
        #print tau,sum1,sum2
        return np.log(sum1)/3-np.log(sum2)/3

    def calc_abTau(self,xdata,ydata,tau):
        return self.calc_aTau(xdata,ydata,tau)-self.calc_bTau(xdata,ydata,tau)

    def calc_sigma12(self,xdata,ydata,alpha):
        dtidata=self.calc_dti(xdata)
        sum1=1e-16
        sum2=1e-16
        sum3=1e-16
        sum4=1e-16
        for i in range(len(xdata)):
            if xdata[i]<alpha:
                sum1+=ydata[i]*dtidata[i]*(xdata[i]-alpha)**2
                sum2+=ydata[i]*dtidata[i]
            else:
                sum3+=ydata[i]*dtidata[i]*(xdata[i]-alpha)**2
                sum4+=ydata[i]*dtidata[i]
        #print xdata[0], alpha, sum1, sum2, sum3, sum4
        return (np.sqrt(sum1/sum2),np.sqrt(sum3/sum4))

    def find_alpha(self,xdata,ydata):
        negi=-1
        posi=-1
        negVal=0
        posVal=0
        for i in range(1,(len(xdata)-1)):
            dum=self.calc_abTau(xdata,ydata,xdata[i])
            #print dum
            if dum<0:
                negi=i
                negVal=dum
            elif negi>0 and dum>0:
                posi=i
                posVal=dum
                break
        dx=xdata[posi]-xdata[negi]
        dy=posVal-negVal
        dydx=dy/dx
        # y-ydata[negi]=dydx*(x-xdata[negi])
        # y=dydx*(x-xdata[negi])+ydata[negi]
        # y=dydx*x+(ydata[negi]-dydx*xdata[negi])
        # 0=dydx*x+(ydata[negi]-dydx*xdata[negi])
        # dydx*x=(dydx*xdata[negi]+ydata[negi])
        alpha=(dydx*xdata[negi]-negVal)/dydx
        #print xdata[negi],xdata[posi],alpha
        return alpha

    def find_delta(self,xdata,ydata,alpha,sigma1,sigma2):
        sum1=1e-16
        sum2=1e-16
        for i in range(len(xdata)):
            zi=self.bigauss([xdata[i]],1,alpha,sigma1,sigma2)[0]
            #print ydata[i],zi,alpha,sigma1,sigma2
            if ydata[i]>0:
                sum1+=np.log(ydata[i]/zi)*zi**2
                sum2+=zi**2
        return np.exp(sum1/sum2)
        

    def find_bigauss(self,xdata,ydata):
        alpha=self.find_alpha(xdata,ydata)
        sigma1,sigma2=self.calc_sigma12(xdata,ydata,alpha)
        delta=self.find_delta(xdata,ydata,alpha,sigma1,sigma2)
        return [delta,alpha,sigma1,sigma2]