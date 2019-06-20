# -*- coding: UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle
import numpy as np

"""
加载数据集
"""


def createDataSet(filename):
    data = open(filename).readlines()
    data_set = []
    for line in data:
        format_line = line.strip().split()
        data_set.append(format_line)
    # data_size = len(data)
    # test_data_size = data_size - train_size
    # train_data, test_data = train_test_split(data_set, test_size=test_data_size / data_size)  # 测试集所占的比例
    return data_set


"""
函数说明:按照给定特征划分数据集

Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
"""


def splitDataSet(dataSet, axis, value):
    retDataSet = []										#创建返回的数据集列表
    for featVec in dataSet: 							#遍历数据集
    	if featVec[axis] == value:
    		reducedFeatVec = featVec[:axis]				#去掉axis特征
    		reducedFeatVec.extend(featVec[axis+1:]) 	#将符合条件的添加到返回的数据集
    		retDataSet.append(reducedFeatVec)
    return retDataSet		  							#返回划分后的数据集


"""
函数说明:统计classList中出现此处最多的元素(类标签)

Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)
"""


def majorityCnt(classList):
    classCount = {}
    for vote in classList:										#统计classList中每个元素出现的次数
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)		#根据字典的值降序排序
    return sortedClassCount[0][0]


"""
函数说明:选择最优特征

Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值
"""

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1					#特征数量
	baseEntropy = calcShannonEnt(dataSet) 				#计算数据集的香农熵
	bestInfoGain = 0.0  								#信息增益
	bestFeature = -1									#最优特征的索引值
	for i in range(numFeatures): 						#遍历所有特征
		#获取dataSet的第i个所有特征
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)     					#创建set集合{},元素不可重复
		newEntropy = 0.0  								#经验条件熵
		for value in uniqueVals: 						#计算信息增益
			subDataSet = splitDataSet(dataSet, i, value) 		#subDataSet划分后的子集
			prob = len(subDataSet) / float(len(dataSet))   		#计算子集的概率
			newEntropy += prob * calcShannonEnt(subDataSet) 	#根据公式计算经验条件熵
		infoGain = baseEntropy - newEntropy 					#信息增益
		# print("第%d个特征的增益为%.3f" % (i, infoGain))			#打印每个特征的信息增益
		if (infoGain > bestInfoGain): 							#计算信息增益
			bestInfoGain = infoGain 							#更新信息增益，找到最大的信息增益
			bestFeature = i 									#记录信息增益最大的特征的索引值
	return bestFeature

""" 
计算数据集的熵
"""


def calcShannonEnt(dataSet):
	numEntires = len(dataSet)						#返回数据集的行数
	labelCounts = {}								#保存每个标签(Label)出现次数的字典
	for featVec in dataSet:							#对每组特征向量进行统计
		currentLabel = featVec[-1]					#提取标签(Label)信息
		if currentLabel not in labelCounts.keys():	#如果标签(Label)没有放入统计次数的字典,添加进去
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1				#Label计数
	shannonEnt = 0.0								#经验熵(香农熵)
	for key in labelCounts:							#计算香农熵
		prob = float(labelCounts[key]) / numEntires	#选择该标签(Label)的概率
		shannonEnt -= prob * log(prob, 2)			#利用公式计算
	return shannonEnt								#返回经验熵(香农熵)


"""
构建决策树
"""


def createTree(dataSet, labels, featLabels):
    label_list = [entry[-1] for entry in dataSet]
    if label_list.count(label_list[0]) == len(label_list):  # 如果所有的数据都属于同一个类别，则返回该类别
        return label_list[0]
    if len(dataSet[0]) == 1:     # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(label_list)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
    del (labels[bestFeat])  # 删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)  # 去掉重复的属性值
    for value in uniqueVals:  # 遍历特征，创建决策树。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


"""
函数说明:使用决策树分类

Parameters:
    myTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testVec - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
"""

def autoNorm(dataSet):
	#获得数据的最小值
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	#最大值和最小值的范围
	ranges = maxVals - minVals
	#shape(dataSet)返回dataSet的矩阵行列数
	normDataSet = np.zeros(np.shape(dataSet))
	#返回dataSet的行数
	m = dataSet.shape[0]
	#原始值减去最小值
	normDataSet = dataSet - np.tile(minVals, (m, 1))
	#除以最大和最小值的差,得到归一化数据
	normDataSet = normDataSet / np.tile(ranges, (m, 1))
	#返回归一化数据结果,数据范围,最小值
	return normDataSet, ranges, minVals


def classify(myTree, featLabels, testVec):
    firstStr = next(iter(myTree))  # 获取决策树结点
    secondDict = myTree[firstStr]  # 下一个字典
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    filename = "page-blocks.data"
    dataSet = createDataSet(filename)
    attribute_label = ['HEIGHT', 'LENGTH', 'AREA', 'ECCEN', 'P_BLACK', 'P_AND', 'MEAN_TR', 'BLACKPIX', 'BLACKAND',
                       'WB_TRANS']
    featLabels = []
    decision_tree = createTree(dataSet, attribute_label, featLabels)
    # 递归会改变attribute_label的值，此处再传一次
    attribute_label = ['HEIGHT', 'LENGTH', 'AREA', 'ECCEN', 'P_BLACK', 'P_AND', 'MEAN_TR', 'BLACKPIX', 'BLACKAND',
                       'WB_TRANS']

    # 取所有数据的百分之十
    hoRatio = 0.10
    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(dataSet)
    # 获得normMat的行数
    m = normMat.shape[0]
    # 百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :],
                                     attribute_label[numTestVecs:m], 4)
        print("分类结果:%s\t真实类别:%s" % (classifierResult, attribute_label[i]))
        if classifierResult != attribute_label[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount / float(numTestVecs) * 100))