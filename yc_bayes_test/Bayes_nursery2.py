import numpy as np
from numpy import *
import random
import re
from collections import Counter
from sklearn.cross_validation import train_test_split

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
	dataSet - 整理的样本数据集
Returns:
	vocabSet - 返回不重复的词条列表，也就是词汇表

"""
def createVocabList():#属性值有重复的 不能合并创建，手动创建
    vocabSet = ['usual','pretentious','great_pret',                     #parents
                'proper','less_proper','improper','critical','very_crit',#has_nurs
                'complete','completed','incomplete','foster',            #form
                '1','2','3','more',                                      #children               
                'convenient','less_conv','critical',                     #housing
                'convenient','inconv',                                   #finance
                'nonprob','slightly_prob','problematic',                #social
                'recommended','priority','not_recom' ]                    #health

    return vocabSet

"""
def loadDataSet：创建实验样本
"""
def loadDataSet(filename):#创建实验样本
    with open(filename,'r',encoding='utf-8') as f:
        dataSet = [data.strip().split(',') for data in f.readlines()]
    classLabel = []
    for line in dataSet:
        classLabel.append(line[-1])
    return dataSet,classLabel
"""

函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters:
	vocabList - createVocabList创建的词条列表
	inputSet - 输入的数据集createVocabList创建的词条列表
Returns:
	returnVec - 文档向量,词集模型

"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(inputSet)
    n=len(vocabList)									#创建一个其中所含元素都为0的向量
    for i in range(n):		
        										#遍历每个词条
        if vocabList[i]=='convenient'and i==4:											#如果词条存在于词汇表中，则置1
            returnVec[16] = 1
        elif vocabList[i]=='convenient'and i==5:
            returnVec[19]=1
        elif vocabList[i]=='critical'and i==1:
            returnVec[6]=1
        elif vocabList[i]=='critical'and i==4:
            returnVec[18]=1
        
        else :
            returnVec[inputSet.index(vocabList[i])] = 1
        
       # else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec													#返回文档向量

"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
	trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	pVect - 各种情况的条件概率
	pClass - 各种结果的概率
	

"""

def trainNB0(trainMatrix,trainCategory):#训练部分
    numTrainDocs = len(trainMatrix)							#计算训练的文档数目
    numWords = len(trainMatrix[0])	
    pClass=[0.0,0.0,0.0,0.0,0.0]						#计算每篇文档的词条数
    '''
    result=Counter(trainCategory)
    pClass[0]=result['not_recom']/float(numTrainDocs)   #计算每种情况概率
    pClass[1]=result['recommend']/float(numTrainDocs)
    pClass[2]=result['very_recom']/float(numTrainDocs)
    pClass[3]=result['priority']/float(numTrainDocs)
    pClass[4]=result['spec_prior']/float(numTrainDocs)
    print(pClass)
    '''
    pClass[0] = 0.33333; pClass[1]=0.00015;
    pClass[2] = 0.02531; pClass[3] = 0.32917; pClass[4] = 0.31204
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p2Num = np.ones(numWords); p3Num = np.ones(numWords);
    p4Num = np.ones(numWords)	#创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 5.0; p1Denom = 5.0 #分母初始化为5,拉普拉斯平滑
    p2Denom = 5.0; p3Denom = 5.0; p4Denom=5.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 'not_recom':							                                               
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        elif trainCategory[i] == 'recommend':												#统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        elif trainCategory[i] == 'very_recom':												#统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p2Num += trainMatrix[i]
            p2Denom += sum(trainMatrix[i])
        elif trainCategory[i] == 'priority':												#统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p3Num += trainMatrix[i]
            p3Denom += sum(trainMatrix[i])
        elif trainCategory[i] == 'spec_prior':												#统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p4Num += trainMatrix[i]
            p4Denom += sum(trainMatrix[i])
    pVect=[]   
    pVect.append(np.log(p0Num/p0Denom))
    pVect.append(np.log(p1Num/p1Denom))
    pVect.append(np.log(p2Num/p2Denom)) 
    pVect.append(np.log(p3Num/p3Denom))						#取对数，防止下溢出          拉普拉斯修正
    pVect.append(np.log(p4Num/p4Denom))         
    return pVect,pClass							#返回属于各类条件概率数组，文档属于各类的概率

"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
	vec2Classify - 待分类的词条数组
	pVec - 条件概率数组
	pClass - 每一条属于各类的概率
Returns:
	0 - 属于非侮辱类
	1 - 属于侮辱类

"""
def classifyNB(vec2Classify, pVec, pClass):
    p0 = sum(vec2Classify * pVec[0]) + np.log(pClass[0])    	#对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p1 = sum(vec2Classify * pVec[1]) + np.log( pClass[1])
    p2 = sum(vec2Classify * pVec[2]) + np.log( pClass[2])
    p3 = sum(vec2Classify * pVec[3]) + np.log( pClass[3])
    p4 = sum(vec2Classify * pVec[4]) + np.log( pClass[4])
    result=max(p0,p1,p2,p3,p4)
    if result == p0:
        return 'not_recom'
    elif result == p1:
        return 'recommend'
    elif result == p2:
        return 'very_recom'
    elif result == p3:
        return 'priority'
    else:
        return 'spec_prior'

"""
函数说明:测试朴素贝叶斯分类器

"""
def nurseryTest():
    dataSet,classLabel=loadDataSet('nursery.data')
    trainSet,testSet=train_test_split(dataSet,test_size=0.1)
    vocabList=createVocabList()
    
    trainSetAttr=[]
    trainSetResult=[]
    for data in trainSet:#训练集部分
        trainSetAttr.append(data[:-1])#前八个属性
        trainSetResult.append(data[-1])#分类结果
    
    testSetAttr=[]#测试集部分
    testSetResult=[]
    for data in testSet:
        testSetAttr.append(data[:-1])#前八个属性
        testSetResult.append(data[-1])#分类结果

    trainMat=[]
    for data in trainSetAttr:
        trainMat.append(setOfWords2Vec(data,vocabList))#构造训练集矩阵
    pVect,pClass=trainNB0(array(trainMat),array(trainSetResult))#训练得到的朴素贝叶斯分类器
    testNum=len(testSet)
    ErrorCount=0
    for i in range(testNum):
        testdata=testSetAttr[i]
        testVec=array(setOfWords2Vec(testdata,vocabList))
        if classifyNB(testVec,pVect,pClass)!=testSetResult[i]:
            ErrorCount+=1
            print("分类错误的测试集：",testSet[i])
    print('正确率：%.2f%%' % (100-(float(ErrorCount) / len(testSet) * 100)))
         

if __name__ == '__main__':
    nurseryTest()