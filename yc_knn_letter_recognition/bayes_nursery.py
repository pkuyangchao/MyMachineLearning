import numpy as np
from functools import reduce
"""
加载数据集
Return:
retrianList:测试集
retesttrain：训练集
"""


def loadDataset (filename):

    data = open(filename, 'r', encoding='utf-8').readlines()
    data[0] = data[0].lstrip('\ufeff')
    dataList = []
    classVec = []
    for line in data:
        format_line = line.strip().split(",")
        dataList.append(format_line[0:8])
        classVec.append(format_line[-1])

    m = len(dataList)
    a = int(0.1 * m)
    x = m - a
    retrianList = dataList[:x]
    retesttrain = dataList[x:]
    return retrianList, retesttrain, classVec


"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
	dataSet - 整理的样本数据集
Returns:
	vocabSet - 返回不重复的词条列表，也就是词汇表
1
"""


def createVocabList(dataSet):
    vocabSet = set([])  					#创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)



"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词集模型

"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)									#创建一个其中所含元素都为0的向量
    for word in inputSet:												#遍历每个词条
        if word in vocabList:											#如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec													#返回文档向量


"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
	trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:

"""


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数

    num_not_recom = 0
    num_recommend = 0
    num_very_recom = 0
    num_priority = 0
    num_spec_prior = 0

    for word in trainCategory:
        if word == "not_recom":
            num_not_recom += 1
        elif word == "recommend":
            num_recommend += 1
        elif word == "very_recom":
            num_very_recom += 1
        elif word == "priority":
            num_priority += 1
        elif word == "spec_prior":
            num_spec_prior += 1

    # 文档属于某类的概率
    p0 = num_not_recom/float(numTrainDocs)
    p1 = num_recommend/float(numTrainDocs)
    p2 = num_very_recom/float(numTrainDocs)
    p3 = num_priority/float(numTrainDocs)
    p4 = num_spec_prior/float(numTrainDocs)

    #词条出现数初始化为1，拉普拉斯平滑, 注意p0Num - p4Num和上面类对应

    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p2Num = np.ones(numWords)
    p3Num = np.ones(numWords)
    p4Num = np.ones(numWords)

    p0Denom = 2.0
    p1Denom = 2.0                        	#分母初始化为0.0,分母初始化为2,拉普拉斯平滑
    p2Denom = 2.0
    p3Denom = 2.0
    p4Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == "not_recom":
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "recommend":
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "very_recom":
            p2Num += trainMatrix[i]
            p2Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "priority":
            p3Num += trainMatrix[i]
            p3Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "spec_prior":
            p4Num += trainMatrix[i]
            p4Denom += sum(trainMatrix[i])
    p0Vect = np.log(p0Num / p0Denom)					#相除
    p1Vect = np.log(p1Num/p1Denom)
    p2Vect = np.log(p2Num/p2Denom)
    p3Vect = np.log(p3Num/p3Denom)
    p4Vect = np.log(p4Num/p4Denom)
    return p0Vect, p1Vect, p2Vect, p3Vect, p4Vect, p0, p1, p2, p3, p4			#返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率


"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
	vec2Classify - 待分类的词条数组
	pXVec - X类的条件概率数组
	pX - 文档属于X类的概率
Returns:


"""


def classifyNB(vec2Classify, p0Vec, p1Vec, p2Vec, p3Vec, p4Vec, p0, p1, p2, p3, p4):
    # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(p0)
    p1 = sum(vec2Classify * p1Vec) + np.log(p1)
    p2 = sum(vec2Classify * p2Vec) + np.log(p2)
    p3 = sum(vec2Classify * p3Vec) + np.log(p3)
    p4 = sum(vec2Classify * p4Vec) + np.log(p4)
    p = max(p0, p1, p2, p3, p4)
    if p == p0:
        return "not_recom"
    elif p == p1:
        return "recommend"
    elif p == p2:
        return "very_recom"
    elif p == p3:
        return "priority"
    elif p == p4:
        return "spec_prior"


"""
函数说明:测试朴素贝叶斯分类器
"""


def testingNB():
    listOPosts, testEntry, listClasses = loadDataset("nursery.data")									#创建实验样本
    myVocabList = createVocabList(listOPosts)								#创建词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))				#将实验样本向量化
    p0V, p1V, p2V, p3V, p4V, p0, p1, p2, p3, p4 = trainNB0(np.array(trainMat), np.array(listClasses))		#训练朴素贝叶斯分类器

    # 分类错误计数
    errorCount = 0.0
    i = len(listOPosts)
    for test in testEntry:
        thisDoc = np.array(setOfWords2Vec(myVocabList, test))  # 测试样本向量化
        result = classifyNB(thisDoc, p0V, p1V, p2V, p3V, p4V, p0, p1, p2, p3, p4)
        print("分类结果:%s\t真实类别:%s" % (result, listClasses[i]))
        if result != listClasses[i]:
            errorCount += 1.0
        i += 1
    print("错误率:%f%%" % (errorCount / float(int(len(testEntry))) * 100))


if __name__ == '__main__':
    testingNB()