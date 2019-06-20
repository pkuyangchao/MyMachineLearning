# -*- coding: UTF-8 -*-
from numpy import *
import numpy as np
from numpy import linalg as la



def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split(',')) - 1
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split(',')
        for i in range(numFeat):
            lineArr.append(curLine[i])
        dataMat.append(lineArr)

    return dataMat


def transform(dataMat):
    count=0
    length=len(dataMat)
    for i in range(30000):
        if dataMat[i][0]=='C':
            count+=1
    dataSet=[[0 for col in range(300)] for row in range(count)]
    j=-1
    for i in range(30000):
        if dataMat[i][0]=='C':
            j+=1
        elif dataMat[i][0]=='V':
            index=int(dataMat[i][1])-1000
            dataSet[j][index]=1
        else:
            continue
    return dataSet


def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def svdEst(dataMat, user, simMeas, item):
    dataMat=mat(dataMat)
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,V = la.svd(dataMat)
    arr=np.array(Sigma)
    total=arr.sum()
    arr=arr/total*100
    print(arr)
    Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def recommend(dataMat, user,N=3, simMeas=ecludSim, estMethod=svdEst):
    dataMat=mat(dataMat)
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#find unrated items 
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def sigmaweight(sigma):
    arr=np.array(sigma)
    total=arr.sum()
    arr=arr/total*100
    print(arr)


if __name__=="__main__":
    dataMat=loadDataSet('anonymous-msweb.data')
    dataSet=transform(dataMat)
    #similarity,sigma=svdEst(dataSet, 3, ecludSim, 3)
    max=recommend(dataSet, 3,3, ecludSim, svdEst)
    print(max)
    
    
    
             