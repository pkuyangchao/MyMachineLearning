# -*- coding:utf-8 -*-
import numpy as np
import random


"""
切分测试集和训练集
参数：
    trainingSet：数据集
    trainingLabels：标签集
    train_size：切分比例（0-1）
返回：
    x_train：训练集
    x_test：测试集
    y_train：训练集标签
    y_test：测试集标签
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

"""
加载数据集
"""
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1  # 行长
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')  # 每一行数据集
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat  # 返回的是list


"""
计算回归系数
参数：
    dataMat:测试集
    labelMat：标签集
返回：回归系数
"""

def standRegres(dataMat, labelMat):
    xMat = np.mat(dataMat)
    yMat = np.mat(labelMat).T  # 把list转换成矩阵形式
    xTx = xMat.T * xMat  # 根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:  # 矩阵求行列式   xTx是一个方阵
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * xMat.T * yMat  # 修改xTx.I * （xMat.T*yMat） ws为列向量，与yMat相同形式
    return ws  # 返回的是ws权重 为一列


if __name__ == '__main__':
    abX, abY = loadDataSet('2014 and 2015 CSM dataset.txt')
    x_train, x_test, y_train, y_test = trainTestSplit(abX, abY, 0.2)

    ws = standRegres(x_train, y_train)
    yHat = np.mat(x_test) * ws
    #  	print(int(yHat[1]))
    yHat = np.around(yHat, decimals=1)
    add = 0
    for i in range(len(x_test)):
        temp = yHat[i] - y_test[i]
        if temp == 0:
            add += 1
    print("与真实ratings相等：Acurracy1={}".format(add / 46))

    add2 = 0
    for i in range(len(x_test)):
        temp = yHat[i] - y_test[i]
        if -1 <= temp <= 1:
            add2 += 1
    print("与真实ratings相差正负一：Acurracy2={}".format(add2 / 46))
