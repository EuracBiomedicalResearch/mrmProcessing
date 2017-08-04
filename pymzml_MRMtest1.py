import pymzml
import csv
from collections import OrderedDict
#import decimal
import subprocess
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import scipy as sci
from scipy import signal
from scipy import linalg
from scipy.signal import argrelextrema

from timeit import default_timer as timer

import kde
import datetime
import readMRM
import glob
import progressbar



folder='./ss2017_07_27/'
folder='/home/ssmarason/mzMLdata/qtrap/CHRIS5000/KIT2-0-5614/2017-06-27/calqc'
folder='/home/ssmarason/mzMLdata/qtrap/CHRIS5000/KIT2-0-5614/2017-06-15/KIT2-LC_5614'
f=glob.glob(os.path.join(folder,'*_4373*.mzML')) #SS_170615



def readMQresults(mqFilename):
    with open(mqFilename) as fmq:
        reader = csv.DictReader(fmq, delimiter='\t')
        for s in reader:
            aPath=s['Original Filename'][:-11]
            aPath=aPath.replace(":\\","/")
            aPath=aPath.replace("\\","/")
            aPath=os.path.split(aPath)
            aPath=aPath[1][:-5]
            #print aPath
            if aPath not in mqRes:
                mqRes[aPath]=dict()
                mqResConc[aPath]=dict()
            if s['IS']=='False' and s['Used']=='True':
                allComps[s['Component Name']]=s['IS Name']
                try:
                    mqRes[aPath][s['Component Name']]=float(s['Area Ratio'])
                except:
                    mqRes[aPath][s['Component Name']]="N/A"
                try:
                    mqResConc[aPath][s['Component Name']]=float(s['Actual Concentration'])
                except:
                    mqResConc[aPath][s['Component Name']]=0
                # 'Retention Time'
    fmq.close()
mqRes=dict()
mqResConc=dict()
allComps=dict()
readMQresults('1023179886_MQ.csv')

mrmImportTemplate='MRM_import_template.csv'
mrmDict=OrderedDict()
with open(mrmImportTemplate) as mrmf:
    myTemplate=mrmf.read().splitlines()
    for oneLine in myTemplate:
        splitLine=oneLine.split("\t")
        mrmDict[splitLine[3]]=splitLine
mrmf.close()



#myRunName = 'KIT2-0-5614_1023182405_26_0_1_2_04_721046-1023182405_1_2_04_721046.mzML'
#print f
#print 'plate','sampl','well','inj','samplType',
#print 'compound','gauss','intens','rt','sd','bigauss','intens','rt','sd1','sd2',
#print 'area1','area2'
resultDic=dict()
ratioDict=dict()
cnt=0
totCnt=len(f)
#print 'starting'
test=readMRM.myLCMSrun('LC_RT_Biocrates2.csv')
#print 'done initiating'
#progressbar.progressbar(cnt, totCnt, prefix = 'Data processing:', suffix = 'Complete')
profiles=dict()
   
rtDict=dict() 
for oneF in f:
    #print oneF
    #oneF.find('_7210')!=-1 or 
    if 1==1: #oneF.find('_437301')!=-1: #or oneF.find('_7210')!=-1: # and oneF.find('10000002')==-1:
        #
        aPath=os.path.split(oneF)
        aPath=aPath[1][:-5]
        aPath=aPath[:aPath.rfind('-')]
        print aPath
        #print oneF
        #fname=fname[:-5].split("-")[0].split("_")
        #plate=fname[0]
        #sampl=fname[6]
        #well=fname[1]
        #inj=fname[4]
        #samplType=fname[5]
        #print 'reading', oneF
        #print 'reading', oneF
        #print oneF
        test.readData(oneF)
        #print 'analyzing'
        resultDic[aPath]=test.fitalldata(showPlot=False,makeNew=True)
        ratioDict[aPath]=test.calcratios()
        for oneName, oneRes in resultDic[aPath].iteritems():
            if oneName not in rtDict:
                rtDict[oneName]=list()
            rtDict[oneName].append({'rtgauss':oneRes[1][1], 'rtbigauss':oneRes[3][1],'file':oneF})
        
    cnt+=1
    #progressbar.progressbar(cnt, totCnt, prefix = 'Data processing:', suffix = 'Complete')

#f1.close()

# allComps2=test.calcratios().keys()
# for oneComp in allComps2:
#     gaussRT=[x['rtgauss'] for x in rtDict[oneComp]]
#     bigaussRT=[x['rtbigauss'] for x in rtDict[oneComp]]
#     myMed=np.median(gaussRT)
#     myAve=np.mean(gaussRT)
#     mySdev=np.std(gaussRT)
#     if abs(myMed-myAve)/myMed>0.001 or mySdev/myMed>0.002:
#         print "RT warning for: ",oneComp, round(myMed,4), round(myAve,4), round(mySdev,6), abs(myMed-myAve)/myMed, mySdev/myMed
#     mrmDict[oneComp][2]=round(myMed,2)
#     IScomp=allComps[oneComp]
#     gaussRT=[x['rtgauss'] for x in rtDict[IScomp]]
#     bigaussRT=[x['rtbigauss'] for x in rtDict[IScomp]]
#     myMed=np.median(gaussRT)
#     myAve=np.mean(gaussRT)
#     mySdev=np.std(gaussRT)
#     mrmDict[IScomp][2]=round(myMed,2)

# mrmImport="MRM_SS_rt_update_"+datetime.date.today().strftime("%Y_%m_%d")+".csv"
# mrmFile=open(mrmImport,'w') 
# for oneComp in mrmDict:
#     mrmFile.write("\t".join([str(x) for x in mrmDict[oneComp]])+"\n")
# mrmFile.close()

oneMRM=test.allMRMs[0]
correlDict=dict()
correlDict['mq']=list()
correlDict['gauss']=list()
correlDict['bigauss']=list()

def calcR2(xdata,ydata,myFunction):
    yhat = myFunction(xdata)                       
    ybar = np.sum(ydata)/len(ydata)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((ydata-ybar)**2)
    return ssreg / sstot

def fitWLS(xdata,ydata,polydegree,w=np.array([])):
    if w.size==0:
        w=np.ones(xdata.size)
    xmatrix=list()
    for oneX in xdata:
        for i in range(polydegree,-1,-1):
            xmatrix.append(np.power(oneX,i))
    xmatrix=np.array(xmatrix)
    xmatrix.shape=(xdata.size,polydegree+1)
    xdataw= xmatrix * np.sqrt(w)[:,None]
    ydataw= ydata * np.sqrt(w)
    return linalg.lstsq(xdataw,ydataw)

def calcRoots(pfun,y0,miny,maxy):
    myRoots=(pfun - y0).roots
    possibles=[]
    miny=miny*.8
    maxy=maxy*1.2
    for oneRoot in myRoots:
        if oneRoot >=miny and oneRoot<=maxy:
            possibles.append(oneRoot)
    if len(possibles)==1:
        return possibles[0]
    elif len(possibles)==2:
        return -1
    else:
        return -2

poly_bigauss=[]
poly_gauss=[]
poly_mq=[]
poly_minmax=[]

for oneComp in allComps:
    #oneComp=allComps[i]
    bigauss=list()
    gauss=list()
    mq=list()
    conc=list()

    qcbigauss=list()
    qcgauss=list()
    qcmq=list()
    qcconc=list()

    # plt.title('Combined profiles for '+oneComp)
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Ion counts (au)')

    for oneFile,oneRatio in ratioDict.iteritems():
        #print oneFile
        if oneFile.find('_4373')!=-1:
            #print 'cal file'
            if oneComp in mqResConc[oneFile]:
                #print ratioDict[oneFile][oneComp]['bigauss']
                if ratioDict[oneFile][oneComp]['bigauss']>0 and ratioDict[oneFile][oneComp]['gauss']>0:
                    bigauss.append(ratioDict[oneFile][oneComp]['bigauss'])
                    gauss.append(ratioDict[oneFile][oneComp]['gauss'])
                    mq.append(float(mqRes[oneFile][oneComp])) #area ratio
                    conc.append(float(mqResConc[oneFile][oneComp])) #conc
        elif oneFile.find('_7210')!=-1:
            if oneComp in mqResConc[oneFile]:
                qcbigauss.append(ratioDict[oneFile][oneComp]['bigauss'])
                qcgauss.append(ratioDict[oneFile][oneComp]['gauss'])
                qcmq.append(float(mqRes[oneFile][oneComp])) #area ratio
                qcconc.append(float(mqResConc[oneFile][oneComp])) #conc


    # mqline=sci.stats.linregress(np.array(conc),np.array(mq))
    # bigausline=sci.stats.linregress(np.array(conc),np.array(bigauss))
    # gausline=sci.stats.linregress(np.array(conc),np.array(gauss))

    # correlDict['mq'].append(mqline[2])
    # correlDict['gauss'].append(gausline[2])
    # correlDict['bigauss'].append(bigausline[2])
    #print oneComp
    #print conc
    #print mq
    #print gauss
    #print bigauss
    conc=np.array(conc)
    mq=np.array(mq)
    bigauss=np.array(bigauss)
    gauss=np.array(gauss)
    weight=1/np.square(conc) #np.square(conc)
    #weight=weight/max(weight)
    mqpoly=fitWLS(conc,mq,2,w=weight)[0]
    poly_mq.append(np.poly1d(mqpoly))
    mqpolyR=calcR2(conc,mq,np.poly1d(mqpoly))

    bigauspoly=fitWLS(conc,bigauss,2,w=weight)[0]
    poly_bigauss.append(np.poly1d(bigauspoly))
    bigauspolyR=calcR2(conc,bigauss,np.poly1d(bigauspoly))
    
    gauspoly=fitWLS(conc,gauss,2,w=weight)[0]
    poly_gauss.append(np.poly1d(gauspoly))
    gauspolyR=calcR2(conc,gauss,np.poly1d(gauspoly))

    correlDict['mq'].append(mqpolyR)
    correlDict['gauss'].append(gauspolyR)
    correlDict['bigauss'].append(bigauspolyR)

    poly_minmax.append((min(conc),max(conc)))
    if oneComp in ['Dopamine']:
    
        minx=min(conc)
        maxx=max(conc)

        plt.plot(conc,mq,'g+',label='mq')
        plt.plot(conc,bigauss,'*',label='bigauss')
        plt.plot(conc,gauss,'.',label='gauss')
        
        p=np.poly1d(bigauspoly)
        xdata=np.arange(minx,maxx*1.05,step=(maxx-minx)/20)
        plt.plot(xdata,p(xdata),label='bigauss fit')
        plt.plot(qcconc,qcbigauss,'r+',label='QCs bigauss')

        p=np.poly1d(mqpoly)
        plt.plot(xdata,p(xdata),label='mq fit')
        plt.plot(qcconc,qcmq,'b+',label='QCs mq')
        plt.legend()
        plt.title(oneComp)
        plt.show()
        continue
    

    # plt.legend()
    # plt.show()

   
    
    # plt.title('Combined bigaussians for '+oneComp)
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Ion counts (au)')
    # for oneFile,oneRes in resultDic.iteritems():
    #     oneRes=
    #     rt=resultDic[oneFile][oneComp][3][1]
    #     sdev=max(resultDic[oneFile][oneComp][3][2:3])
    #     xdata=np.arange(rt-4*sdev,rt+4*sdev,step=0.01)
    #     ydata=oneMRM.bigauss(xdata,*resultDic[oneFile][oneComp][3])
    #     print rt-4*sdev,rt+4*sdev,len(xdata),len(ydata)
    #     labl=str(round(rt*60,1))+':'+oneFile[(oneFile.rfind('_')+1):]
    #     plt.plot((xdata-rt)*60,ydata/resultDic[oneFile][oneComp][3][0],label=labl)
    # plt.legend()
    # plt.show()

minY=min((min(correlDict['mq']),min(correlDict['gauss']),min(correlDict['bigauss'])))
#minY=0.9
cats=3
barwidth=0.35
betweenbarwidt=0.05
betweencatwidth=cats*barwidth+(cats+2)*betweenbarwidt
plt.title('Calibration correlations')
plt.xlabel('Compound')
plt.ylabel('r squared')
ticks=allComps
xrang=np.array(range(len(ticks)))*betweencatwidth
#print len(xrang), len(correlDict['mq'])
plt.bar(xrang,correlDict['mq'],barwidth-betweenbarwidt,label='mq')
plt.bar(xrang+barwidth,correlDict['gauss'],barwidth-betweenbarwidt,label='gauss')
plt.bar(xrang+2*barwidth,correlDict['bigauss'],barwidth-betweenbarwidt,label='bigauss')
plt.xticks(xrang+barwidth, ticks,rotation='vertical')
plt.ylim([minY-0.005,1.0])
plt.legend()
plt.show()




# fout='out.batch170615.arearatio.txt'
# with open(fout,'w') as f:
#     f.write("\t".join(('sample','compound','gauss','bigauss','multiq','gauss conc','bigauss conc','multiq conc'))+"\n")
#     for fname,oneRes in ratioDict.iteritems():
#         for i in range(len(allComps)):
#             oneComp=allComps[i]
#             if oneComp in oneRes:
#                 oneRes2=oneRes[oneComp]
#         #for compound,oneRes2 in oneRes.iteritems():
#                 #if fname in mqResConc:
#                 if oneComp in mqResConc[fname]:
#                     f.write("\t".join((fname, oneComp, str(oneRes2['gauss']), str(oneRes2['bigauss']), str(mqRes[fname][oneComp]))))
#                     mqr=calcRoots(poly_mq[i],mqRes[fname][oneComp],poly_minmax[i][0],poly_minmax[i][1])
#                     bigr=calcRoots(poly_bigauss[i],oneRes2['bigauss'],poly_minmax[i][0],poly_minmax[i][1])
#                     gaur=calcRoots(poly_gauss[i],oneRes2['gauss'],poly_minmax[i][0],poly_minmax[i][1])
#                     #print gaur,bigr,mqr
#                     f.write("\t")
#                     f.write("\t".join((str(gaur),str(bigr),str(mqr))))
#                     f.write("\n")
#         #     oneMRM=test.allMRMs[i]
#         #     if 1==1: #oneMRM.name in showNames:#(oneF.find('10000001.mzML')==-1 and oneF.find('11000002.mzML')==-1) or oneMRM.name in showNames:
#         #         res=oneMRM.makeGaussFit(showPlot=False, spanAll=False, smooth=True, smoothparam=[5,1,'savitskygol'])
#         #         gaussA=res[1][0]
#         #         gaussRt=res[1][1]
#         #         gaussSigma=res[1][2]

#         #         bigaussA=res[3][0]
#         #         bigaussRt=res[3][1]
#         #         bigaussSigma1=res[3][2]
#         #         bigaussSigma2=res[3][3]
#         #         #print '"'+oneF+'"',

                
#         #         # print plate,sampl,well,inj,samplType,
#         #         # print '"'+oneMRM.name+'"',
#         #         # print res[0],gaussA,gaussRt,gaussSigma,
#         #         # print res[2],bigaussA,bigaussRt,bigaussSigma1,bigaussSigma2,
#         #         # print gaussA*gaussSigma/0.3989,
#         #         # print bigaussA*(bigaussSigma1+bigaussSigma2)/0.3989/2
#         #     #if (res[0]==0 or res[2]==0) and oneF.find('10000001.mzML')==-1 and oneF.find('11000002.mzML')==-1:
#         #     #    test.makeGaussFit(i, showPlot=True, spanAll=False, smooth=True, smoothparam=[5,1,'savitskygol'])


# test=readMRM.myLCMSrun(myRunName)
# #test=readMRM.myLCMSrun(os.path.join(folder,myRunName))
# #test.makedKDE()

# for i in np.arange(0, len(test.allMRMs.elemenstList)):
#     #test.makeGaussFit(i, showPlot=True)
#     #oneMRM=test.allMRMs.elemenstList[i]
#     #print oneMRM.name,rt,oneMRM.rt
#     #test.makePVMGfit(i, showPlot=True)
#     if test.allMRMs.elemenstList[i].name in showNames or test.allMRMs.elemenstList[i].name not in showNames:
#         oneMRM=test.allMRMs.elemenstList[i]
#         #res=test.makeGaussFit(i, showPlot=True, spanAll=True, smooth=True, smoothparam=[5,2,'movingaverage'])
#         res=test.makeGaussFit(i, showPlot=False, spanAll=False, smooth=True, smoothparam=[3,1,'savitskygol'])
#         #test.makePVMGfit(i, True)
#         gaussA=res[1][0]
#         gaussRt=res[1][1]
#         gaussSigma=res[1][2]

#         bigaussA=res[3][0]
#         bigaussRt=res[3][1]
#         bigaussSigma1=res[3][2]
#         bigaussSigma2=res[3][3]
#         print '"'+oneMRM.name+'"',res[0],gaussA,gaussRt,gaussSigma,res[2],bigaussA,bigaussRt,bigaussSigma1,bigaussSigma2

# mzList=(50.01,50.0101,55.5,56.6)

# test2=myMZ(mzList[0],1.0,15)
# for oneMz in mzList:
#     print test2.mzFits(oneMz)

# ppmList=[0.1, 0.2]
# for onePPM in ppmList:
#     test=myLCMSrun(myRunName,onePPM)
#     #print len(test.myData), len(test.allMzDict)

#     test.condenseMZ()

#     #print len(test.condensMzDict)
#     fp = open('out_ppm'+str(onePPM)+'.txt', 'w')
#     fp.write('mz\tcnt\n')
#     totCnt=0
#     for idx,oneDict in test.condensMzDict.iteritems():
#         totCnt+=len(oneDict)
#         for idx2, oneMyMZ in oneDict.iteritems():
#             fp.write(str(oneMyMZ.getMed())+'\t'+str(len(oneMyMZ.getmzList()))+'\n')

#     fp.close()
#     print onePPM, totCnt

# divisorList=[100,]#1000,10000,100000]
# for oneDivisor in divisorList:
#     begintime=timer()
#     test=readbkg.myLCMSrun(myRunName,5, timeLimits=(0.0,5.5),divisor=oneDivisor)
#     #print len(test.myData), len(test.allMzDict)
#     totTime=timer()-begintime
#     print 'read',oneDivisor, totTime
#     begintime=timer()
#     test.makeMZDict()
#     totTime=timer()-begintime
#     print 'firstpass',oneDivisor, totTime
#     begintime=timer()
#     res=test.makedKDE()
#     #test.condenseMZ()
#     totTime=timer()-begintime
#     print 'secondpass',oneDivisor, totTime
#     for idx in test.condensMzDict:
#         #if round(idx,0)==91:
#         oneMyMZ=test.condensMzDict[idx]
#         if len(oneMyMZ.getmzList())>100:
#             #print idx, oneMyMZ.getMed()
#             myVals=[oneMz - oneMyMZ.getMed() for oneMz in oneMyMZ.getmzList()]
#             #myVals=oneMyMZ.getmzList()
#             myTime=oneMyMZ.gettimeList()
#             myIntens=oneMyMZ.getintensList()

#             myMed=len(myVals)*[oneMyMZ.getMed()]
#                 #plt.scatter(myMed,myVals,s=1)
#             mmax=max(myIntens)
#             mmin=min(myIntens)
#             mav=(mmax+mmin)/2
#             #if np.log(mmax-mmin)<6:
#             plt.scatter(myTime,myVals,s=1)
#                 #plt.scatter(myTime,np.log(myIntens[:]),s=1)
#                 #plt.scatter(idx,oneMyMZ.getMed())
#     plt.show()

# for oneRes in res:
#     if oneRes is not None:
#         plt.plot(oneRes[1],oneRes[2])
#         plt.scatter(oneRes[3], len(oneRes[3])*[max(oneRes[2])*1.1])
#         #for oneMz in oneRes[3]:
#         #    print oneMz, oneRes[0]
#         #print oneRes[0]
    
# plt.show()

# folder="/home/ssmarason/mzMLdata/MitraStab/2016_07"
# f = open("newMzList2.txt", "w")
# subprocess.call(['find', folder, "-type","f", "-name","*.mzML", '-printf', '%p\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n'],stdout=f)
# #find /var -maxdepth 1 -type d -printf "%p %TY-%Tm-%TdT%TH:%TM:%TS%Tz\n"
# f.close

# allFiles=list()
# with open("newMzList2.txt") as f:
#     for s in f.read().splitlines():
#         dum=s.split("\t")
#         allFiles.append(dum[0])
# f.close()

# oneDivisor=100

# allFiles=list()
# myRunName = os.path.join("/home/ssmarason/mzMLdata/MitraStab/2017_02",'090217_21m_RT_-80_2h_b.mzML')
# allFiles.append(myRunName)
# myRunName =os.path.join("/home/ssmarason/mzMLdata/MitraStab/2016_07",'140716_MITRA_EXT_1B5A_POS.mzML')
# allFiles.append(myRunName)

# for myRunName in allFiles:
#     fileresult=os.path.split(myRunName)[1]+'.res.txt'
#     fileresult=os.path.join('test',fileresult)
#     test=readbkg.myLCMSrun(myRunName,5, timeLimits=(0.0,5.5), divisor=oneDivisor)
#     test.makeMZDict()
#     res=test.makedKDE()
    
#     fp=open(fileresult,'w')
#     for idx in test.condensMzDict:
#         oneMyMZ=test.condensMzDict[idx]
#         for oneMZ in oneMyMZ.getelemList():
#             fp.write(str(idx))
#             fp.write('\t')
#             fp.write(str(oneMZ.getMz()))
#             fp.write('\t')
#             fp.write(str(oneMZ.getTime()))
#             fp.write('\t')
#             fp.write(str(oneMZ.getIntens()))
#             fp.write('\n')
#     fp.close()

# print "End"
    # for oneRes in res:
    #      if oneRes is not None:
    #         plt.plot(oneRes[1],oneRes[2])
    #         plt.scatter(oneRes[3], len(oneRes[3])*[max(oneRes[2])*1.1])
    #         #print oneRes[0]
    
    # plt.show()


# onePPM=1.5
# test=myLCMSrun(myRunName,onePPM)

# test.condenseMZ()
# for idx in test.condensMzDict:
#     if round(idx,0)==91:
#         print idx

# oneDict=test.condensMzDict[90.7]
# i=0
# for idx2, oneMyMZ in oneDict.iteritems():
#     myVals=np.array(oneMyMZ.getmzList())
#     myMed=np.zeros(len(myVals))
#     myMed.fill(oneMyMZ.getMed())
#     plt.scatter(myMed,myVals,s=1)


# plt.show()
