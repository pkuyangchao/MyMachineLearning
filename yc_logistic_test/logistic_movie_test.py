import xlrd
import numpy as np

"""
函数说明：读文件数据
Parameters:
    filename - 读取文件路径
Returns:
    dataSet - 获得的数据集
    labelSet - 获得的标签集
"""


def loadDataSet(filename):
    dataSet = []
    labelSet = []
    # 读取表格文件
    fp = xlrd.open_workbook(filename)
    table = fp.sheets()[0]
    rows = table.nrows
    # 处理文件
    for i in range(1, rows):
        subData = [1.0]
        subData.extend(table.row_values(i)[1:2])
        subData.extend(table.row_values(i)[3:])
        labelSet.append(table.row_values(i)[2])
        dataSet.append(subData)

    return dataSet, labelSet


"""
函数说明：使用梯度下降算法，需对数据进行特征缩放
Parameters:
    dataSet - 进行处理的数据集
Returns:
    dataSet - 处理过后的数据集
"""


def formDataSetGrad(dataSet):
    rows = len(dataSet)
    cols = len(dataSet[0])
    # 将数据中空白部分置为0
    for i in range(rows):
        for j in range(cols):
            if dataSet[i][j] == '':
                dataSet[i][j] = 0.0
    # 将数据集转化为array，便于操作
    dataSet = np.array(dataSet)
    min = dataSet.min(0)
    max = dataSet.max(0)
    average = dataSet.mean(0)
    # 进行特征缩放
    for i in range(rows):
        for j in range(1, cols):
            dataSet[i][j] = float((dataSet[i][j] - average[j]) / (max[j] - min[j]))

    return dataSet


"""
函数说明：使用正规方程法，对数据空白处进行处理
Parameters:
    dataSet - 进行处理的数据集
Returns:
    dataSet - 处理过后的数据集
"""


def formDataSet(dataSet):
    rows = len(dataSet)
    cols = len(dataSet[0])
    # 将数据中空白部分置为0
    for i in range(rows):
        for j in range(cols):
            if dataSet[i][j] == '':
                dataSet[i][j] = 0.0
    # 将数据集转化为array，便于操作
    dataSet = np.array(dataSet)

    return dataSet


"""
函数说明：sigmoid表达式
Parameters:
    inX - 参数当前最优解的
Returns:
    sigmoidNum - 求得的sigmoid大小
"""


def sigmoid(inX):
    sigmoidNum = 1.0 / (1 + np.exp(-inX))
    return sigmoidNum


"""
函数说明：使用logistic回归求解最优参数解
Parameters:
    trainingDataSet - 训练数据集
    labelSet - 训练标签集
Returns:
    weights - 最用参数解
"""


def gradAscent(trainingDataSet, labelSet):
    # 将训练数据集和标签集转化为矩阵，方便运算
    dataMat = np.mat(trainingDataSet)
    labelMat = np.mat(labelSet).transpose()
    m, n = np.shape(dataMat)
    # 设置学习率，最大循环次数
    alpha = 0.001
    maxCircle = 500
    # 初始化最优参数解
    weights = np.ones((n, 1))

    for k in range(maxCircle):
        h = dataMat * weights
        error = h - labelMat
        weights = weights - alpha / m * dataMat.transpose() * error

    return weights.getA()


"""
函数说明：使用正规方程法求解最优参数解
Parameters:
    trainingDataSet - 训练数据集
    labelSet - 训练标签集
Returns:
    weights - 最用参数解
"""


def normalEquation(trainingDataSet, labelSet):
    # 将训练数据集和标签集转化为矩阵，方便运算
    dataMat = np.mat(trainingDataSet)
    labelMat = np.mat(labelSet).transpose()
    # 使用正规方程法求解最优解
    weights = (dataMat.transpose() * dataMat).I * dataMat.transpose() * labelMat

    return weights.getA()


"""
函数说明：对采用梯度下降算法获得参数最优解做测试
Parameters:
    testDataSet - 用于测试的数据集
    testLabelSet - 用于测试的标签集
    weights - 用于测试的最优参数解
Returns:
    无
"""


def gradAscentTest(testDataSet, testLabelSet, weights):
    testDataMat = np.mat(testDataSet)
    predictLabelSet = sigmoid(testDataMat * weights)

    for i in range(len(testLabelSet)):
        error = predictLabelSet[i] - testLabelSet[i]
        print("预测rating：" + str(predictLabelSet[i]) + "；真实rating：" + str(testLabelSet[i]) + "；误差大小：" + str(error))


"""
函数说明：对采用正规方程法获得参数最优解做测试
Parameters:
    testDataSet - 用于测试的数据集
    testLabelSet - 用于测试的标签集
    weights - 用于测试的最优参数解
Returns:
    无
"""


def normalEquationTest(testDataSet, testLabelSet, weights):
    testDataMat = np.mat(testDataSet)
    predictLabelSet = testDataMat * weights
    errorCount = 0

    for i in range(len(testLabelSet)):
        error = predictLabelSet[i] - testLabelSet[i]
        if error < -1 or error > 1:
            errorCount += 1
        print("预测rating：" + str(predictLabelSet[i]) + "；真实rating：" + str(testLabelSet[i]) + "；误差大小：" + str(error))

    errorPercent = float(errorCount / len(testLabelSet)) * 100
    print("预测rating的错误数为：" + str(errorCount) + "，预测错误率为：" + str(errorPercent))


"""
函数说明：通过梯度下降求的的最优参数，或者正规方程法求的最优参数，预测rating
Parameters:
    无
Returns:
    无
"""


def ratingTest():
    # 读取文件，处理数据
    filename = '2014 and 2015 CSM dataset.xlsx'
    dataSet, labelSet = loadDataSet(filename)
    # 采用梯度下降算法，对数据进行预处理，特征缩放
    # dataSet = formDataSetGrad(dataSet)
    # 采用正规方程法，无需特征缩放
    dataSet = formDataSet(dataSet)

    # 提取数据集中的数据，分成两大训练集，CF和SMF
    trainingDataSetCF = []
    trainingDataSetSMF = []
    testDataSetCF = []
    testDataSetSMF = []
    # CF：Genre, Budget, Screens, Sequel；SMF：other 6 attributes
    for i in range(200):
        subDataSetCF = [dataSet[i][0]]
        subDataSetCF.extend(dataSet[i][2:3])
        subDataSetCF.extend(dataSet[i][4:7])
        trainingDataSetCF.append(subDataSetCF)

        subDataSetSMF = [dataSet[i][0]]
        subDataSetSMF.extend(dataSet[i][7:])
        trainingDataSetSMF.append(subDataSetSMF)
    # 提取数据集中剩余部分，作为测试集，分为CF和SMF
    for i in range(200, len(dataSet)):
        subDataSetCF = [dataSet[i][0]]
        subDataSetCF.extend(dataSet[i][2:3])
        subDataSetCF.extend(dataSet[i][4:7])
        testDataSetCF.append(subDataSetCF)

        subDataSetSMF = [dataSet[i][0]]
        subDataSetSMF.extend(dataSet[i][7:])
        testDataSetSMF.append(subDataSetSMF)
    # 提取数据集只能够的标签转化为训练和测试的两部分
    trainingLabelSet = labelSet[:200]
    testLabelSet = labelSet[200:]
    # 运用梯度下降算法，求最优解并测试
    """
    weightsCF = gradAscent(trainingDataSetCF, trainingLabelSet)
    print(weightsCF)
    gradAscentTest(testDataSetCF, testLabelSet, weightsCF)
    weightsSMF = gradAscent(trainingDataSetSMF, trainingLabelSet)
    print(weightsSMF)
    gradAscentTest(testDataSetSMF, testLabelSet, weightsSMF)
    """
    # 使用正规方程法求解最优解并测试
    weightsCF = normalEquation(trainingDataSetCF, trainingLabelSet)
    print(weightsCF)
    normalEquationTest(testDataSetCF, testLabelSet, weightsCF)

    weightsSMF = normalEquation(trainingDataSetSMF, trainingLabelSet)
    print(weightsSMF)
    normalEquationTest(testDataSetSMF, testLabelSet, weightsSMF)


if __name__ == '__main__':
    ratingTest()