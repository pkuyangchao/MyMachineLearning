# -*-coding:utf-8 -*-
import numpy as np
import csv
import random


"""
分割测试集和训练集
"""
def trainTestSplit(trainingSet, trainingLabels, train_size):
    totalNum = int(len(trainingSet))
    trainIndex = list(range(totalNum))#存放训练集的下标
    x_test = []     #存放测试集输入
    y_test = []      #存放测试集输出
    x_train = []    #存放训练集输入
    y_train = []    #存放训练集输出
    trainNum = int(totalNum * train_size) #划分训练集的样本数
    for i in range(trainNum):
        randomIndex = int(random.uniform(0, len(trainIndex)))
        x_test.append(trainingSet[randomIndex])
        y_test.append(trainingLabels[randomIndex])
        del trainIndex[randomIndex]#删除已经放入测试集的下标
    for i in range(totalNum-trainNum):
        x_train.append(trainingSet[trainIndex[i]])
        y_train.append(trainingLabels[trainIndex[i]])
    return x_train, x_test, y_train, y_test


def loadDataSet(fileName):
	dataMat = []; labelMat = []
	csv_file = csv.reader(open('student-mat.csv','r'),delimiter=';')
	for stu in csv_file:
		lineArr = []
		for i in range(len(stu)-1):
			lineArr.append(stu[i])
		dataMat.append(lineArr)
		labelMat.append(stu[len(stu)-1])
	for i in range(len(dataMat)):
		if(dataMat[i][0]=='GP'):
			dataMat[i][0]=1
		else:
			dataMat[i][0]=2
		if(dataMat[i][1]=='F'):
			dataMat[i][1]=1
		else:
			dataMat[i][1]=2
		if(dataMat[i][2]!=''):
			dataMat[i][2]=int(dataMat[i][2])
		if(dataMat[i][3]=='U'):
			dataMat[i][3]=1
		else:
			dataMat[i][3]=2
		if(dataMat[i][4]=='GT3'):
			dataMat[i][4]=5
		else:
			dataMat[i][4]=3
		if(dataMat[i][5]=='A'):
			dataMat[i][5]=1
		else:
			dataMat[i][5]=2
		if(dataMat[i][6]!=''):
			dataMat[i][6]=int(dataMat[i][6])
		if(dataMat[i][7]!=''):
			dataMat[i][7]=int(dataMat[i][7])
		if(dataMat[i][8]=='teacher'):
			dataMat[i][8]=1
		elif(dataMat[i][8]=='health'):
			dataMat[i][8]=2
		elif(dataMat[i][8]=='services'):
			dataMat[i][8]=3
		elif(dataMat[i][8]=='at_home'):
			dataMat[i][8]=4
		else:
			dataMat[i][8]=5
		if(dataMat[i][9]=='teacher'):
			dataMat[i][9]=1
		elif(dataMat[i][9]=='health'):
			dataMat[i][9]=2
		elif(dataMat[i][9]=='services'):
			dataMat[i][9]=3
		elif(dataMat[i][9]=='at_home'):
			dataMat[i][9]=4
		else:
			dataMat[i][9]=5
		if(dataMat[i][10]=='home'):
			dataMat[i][10]=1
		elif(dataMat[i][10]=='reputation'):
			dataMat[i][10]=2
		elif(dataMat[i][10]=='course'):
			dataMat[i][10]=3
		else:
			dataMat[i][10]=4
		if(dataMat[i][11]=='mother'):
			dataMat[i][11]=1
		elif(dataMat[i][11]=='father'):
			dataMat[i][11]=2
		else:
			dataMat[i][11]=3
		if(dataMat[i][12]!=''):
			dataMat[i][12]=int(dataMat[i][12])
		if(dataMat[i][13]!=''):
			dataMat[i][13]=int(dataMat[i][13])
		if(1<=int(dataMat[i][14])<3):
			dataMat[i][14]=int(dataMat[i][14])
		else:
			dataMat[i][14]=4
		if(dataMat[i][15]=='yes'):
			dataMat[i][15]=1
		else:
			dataMat[i][15]=2
		if(dataMat[i][16]=='yes'):
			dataMat[i][16]=1
		else:
			dataMat[i][16]=2
		if(dataMat[i][17]=='yes'):
			dataMat[i][17]=1
		else:
			dataMat[i][17]=2
		if(dataMat[i][18]=='yes'):
			dataMat[i][18]=1
		else:
			dataMat[i][18]=2
		if(dataMat[i][19]=='yes'):
			dataMat[i][19]=1
		else:
			dataMat[i][19]=2
		if(dataMat[i][20]=='yes'):
			dataMat[i][20]=1
		else:
			dataMat[i][20]=2
		if(dataMat[i][21]=='yes'):
			dataMat[i][21]=1
		else:
			dataMat[i][21]=2
		if(dataMat[i][22]=='yes'):
			dataMat[i][22]=1
		else:
			dataMat[i][22]=2
		if(dataMat[i][23]!=''):
			dataMat[i][23]=int(dataMat[i][23])
		if(dataMat[i][24]!=''):
			dataMat[i][24]=int(dataMat[i][24])
		if(dataMat[i][25]!=''):
			dataMat[i][25]=int(dataMat[i][25])
		if(dataMat[i][26]!=''):
			dataMat[i][26]=int(dataMat[i][26])
		if(dataMat[i][27]!=''):
			dataMat[i][27]=int(dataMat[i][27])
		if(dataMat[i][28]!=''):
			dataMat[i][28]=int(dataMat[i][28])
		if(dataMat[i][29]!=''):
			dataMat[i][29]=int(dataMat[i][2])
		if(dataMat[i][30]!=''):
			dataMat[i][30]=int(dataMat[i][30])
		if(dataMat[i][31]!=''):
			dataMat[i][31]=int(dataMat[i][31])
		if(int(labelMat[i])>=10):
			labelMat[i]=1
		else:
			labelMat[i]=-1
	return dataMat, labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
    单层决策树分类函数
    Parameters:
        dataMatrix - 数据矩阵
        dimen - 第dimen列，也就是第几个特征
        threshVal - 阈值
        threshIneq - 标志
    Returns:
        retArray - 分类结果
    """
    retArray = np.ones((np.shape(dataMatrix)[0],1))                #初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0         #如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0         #如果大于阈值,则赋值为-1
    return retArray

def buildStump(dataArr,classLabels,D):
    """
    找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    """
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    minError = float('inf')                                                        #最小误差初始化为正无穷大
    for i in range(n):                                                            #遍历所有特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()        #找到特征中最小的值和最大值
        stepSize = (rangeMax - rangeMin) / numSteps                                #计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:                                          #大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)                     #计算阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#计算分类结果
                errArr = np.mat(np.ones((m,1)))                                 #初始化误差矩阵
                errArr[predictedVals == labelMat] = 0                             #分类正确的,赋值为0
                weightedError = D.T * errArr                                      #计算误差
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:                                     #找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    """
    使用AdaBoost算法提升弱分类器性能
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        numIt - 最大迭代次数
    Returns:
        weakClassArr - 训练好的分类器
        aggClassEst - 类别估计累计值
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)                                            #初始化权重
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)     #构建单层决策树
        # print("D:",D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))         #计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        bestStump['alpha'] = alpha                                          #存储弱学习算法权重
        weakClassArr.append(bestStump)                                      #存储单层决策树
        # print("classEst: ", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)     #计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()                                                        #根据样本权重公式，更新样本权重
        #计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst                                      #计算类别估计累计值
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))     #计算误差
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate)
        if errorRate == 0.0: break                                             #误差为0，退出循环
    return weakClassArr, aggClassEst

def adaClassify(datToClass,classifierArr):
    """
    AdaBoost分类函数
    Parameters:
        datToClass - 待分类样例
        classifierArr - 训练好的分类器
    Returns:
        分类结果
    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):                                        #遍历所有分类器，进行分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)

if __name__ == '__main__':
    dataArr, LabelArr = loadDataSet('student-mat.csv')
    dataArr,testArr,LabelArr,testLabelArr = trainTestSplit(dataArr,LabelArr,0.2)
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
    print(weakClassArr)
    predictions = adaClassify(dataArr, weakClassArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))
    predictions = adaClassify(testArr, weakClassArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
