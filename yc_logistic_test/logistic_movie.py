# -*- coding:UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random
import xlrd

"""
函数说明:加载数据

Parameters:
    无
Returns:
    dataMat - 数据列表
    labelMat - 标签列表
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-28
"""


def loadDataSet():
    labelMat = []
    dataMat = np.array([])  # 创建标签列表
    file_path = '2014 and 2015 CSM dataset.xlsx'
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_name('Sheet1')
    nrows = table.nrows  # 获取总行数
    ncols = table.ncols  # 获取总列数
    # fr = open('testSet.txt')
    oneline = ()  # 打开文件
    for row_index in range(12, nrows):
        if row_index == 121:
            continue
        one_line = table.row_values(row_index, start_colx=3, end_colx=None)

        for line_index in range(3, ncols - 3):
            if one_line[line_index] == '':
                one_line[line_index] = 0.0
        for x in one_line:
            if isinstance(x, str):
                print(type(x), x, row_index, line_index, oneline)
        one_line = [float(x) for x in one_line]

        oneline = np.array(oneline)
        if row_index == 12:
            dataMat = one_line
        else:
            dataMat = np.vstack((dataMat, one_line))
    label_cal = np.array(table.col_values(2, start_rowx=12, end_rowx=None))
    for label_index in range(0, nrows - 12):
        if label_cal[label_index] > 6.5:
            label_cal[label_index] = 1
        else:
            label_cal[label_index] = 0
    label_cal = np.delete(label_cal, 121)
    return dataMat.tolist(), label_cal.tolist()  # 返回


def testDataSet():
    labelTest = []
    dataMat = np.array([])  # 创建标签列表
    file_path = '2014 and 2015 CSM dataset.xlsx'
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_name('Sheet1')
    nrows = table.nrows  # 获取总行数
    ncols = table.ncols  # 获取总列数
    # fr = open('testSet.txt')
    oneline = ()  # 打开文件
    for row_index in range(1, 13):
        one_line = table.row_values(row_index, start_colx=3, end_colx=None)
        for line_index in range(3, ncols - 3):
            if one_line[line_index] == '':
                one_line[line_index] = 0.0

        for x in one_line:
            if isinstance(x, str):
                print(type(x), x, row_index, line_index, one_line)
        one_line = [float(x) for x in one_line]

        one_line = np.array(one_line)
        if row_index == 1:
            dataMat = one_line
        else:
            dataMat = np.vstack((dataMat, one_line))
    label_cal = np.array(table.col_values(2, start_rowx=1, end_rowx=13))
    for label_index in range(0, 12):
        if label_cal[label_index] > 6.5:
            label_cal[label_index] = 1
        else:
            label_cal[label_index] = 0

    return dataMat.tolist(), label_cal.tolist()


"""
函数说明:sigmoid函数

Parameters:
    inX - 数据
Returns:
    sigmoid函数
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-28
"""


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


"""
函数说明:梯度上升算法

Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
    weights_array - 每次更新的回归系数
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-28
"""


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01  # 移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 5000  # 最大迭代次数
    weights = np.ones((n, 1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA(), weights_array  # 将矩阵转换为数组，并返回


"""
函数说明:改进的随机梯度上升算法

Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
    weights_array - 每次更新的回归系数
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-31
"""


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)  # 参数初始化
    weights_array = np.array([])  # 存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex] * weights))  # 选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h  # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            weights_array = np.append(weights_array, weights, axis=0)  # 添加回归系数到数组中
            del (dataIndex[randIndex])  # 删除已经使用的样本
    weights_array = weights_array.reshape(numIter * m, n)  # 改变维度
    return weights, weights_array  # 返回


"""
函数说明:绘制数据集

Parameters:
    weights - 权重参数数组
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-30
"""


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()  # 加载数据集
    dataArr = np.array(dataMat)  # 转换成numpy的array数组
    n = np.shape(dataMat)[0]  # 数据个数
    xcord1 = [];
    ycord1 = []  # 正样本
    xcord2 = [];
    ycord2 = []  # 负样本
    for i in range(n):  # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])  # 1为正样本
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])  # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)  # 绘制正样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)  # 绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')  # 绘制title
    plt.xlabel('X1');
    plt.ylabel('X2')  # 绘制label
    plt.show()


"""
函数说明:绘制回归系数与迭代次数的关系

Parameters:
    weights_array1 - 回归系数数组1
    weights_array2 - 回归系数数组2
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-30
"""


def plotWeights(weights_array1, weights_array2):
    # 设置汉字格式
    font = FontProperties(fname=r"/Users/yifyang/Downloads/simsunttc/simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=9, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w3与迭代次数的关系
    axs[3][0].plot(x1, weights_array1[:, 3])
    axs2_xlabel_text = axs[3][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[3][0].set_ylabel(u'W3', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w4与迭代次数的关系
    axs[4][0].plot(x1, weights_array1[:, 4])
    axs2_xlabel_text = axs[4][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[4][0].set_ylabel(u'W4', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w5与迭代次数的关系
    axs[5][0].plot(x1, weights_array1[:, 5])
    axs2_xlabel_text = axs[5][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[5][0].set_ylabel(u'W5', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w6与迭代次数的关系
    axs[6][0].plot(x1, weights_array1[:, 6])
    axs2_xlabel_text = axs[6][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[6][0].set_ylabel(u'W6', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w7与迭代次数的关系
    axs[7][0].plot(x1, weights_array1[:, 7])
    axs2_xlabel_text = axs[7][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[7][0].set_ylabel(u'W7', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w8与迭代次数的关系
    axs[8][0].plot(x1, weights_array1[:, 8])
    axs2_xlabel_text = axs[8][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[8][0].set_ylabel(u'W8', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W2', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[3][1].plot(x2, weights_array2[:, 3])
    axs2_xlabel_text = axs[3][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[3][1].set_ylabel(u'W3', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w3与迭代次数的关系
    axs[4][1].plot(x2, weights_array2[:, 4])
    axs2_xlabel_text = axs[4][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[4][1].set_ylabel(u'W4', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    plt.show()
    # 绘制w5与迭代次数的关系
    axs[5][1].plot(x2, weights_array2[:, 5])
    axs2_xlabel_text = axs[5][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[5][1].set_ylabel(u'W5', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    plt.show()
    # 绘制w6与迭代次数的关系
    axs[6][1].plot(x2, weights_array2[:, 6])
    axs2_xlabel_text = axs[6][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[6][1].set_ylabel(u'W6', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    plt.show()
    # 绘制w7与迭代次数的关系
    axs[7][1].plot(x2, weights_array2[:, 7])
    axs2_xlabel_text = axs[7][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[7][1].set_ylabel(u'W7', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    plt.show()
    # 绘制w8与迭代次数的关系
    axs[8][1].plot(x2, weights_array2[:, 8])
    axs2_xlabel_text = axs[8][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[8][1].set_ylabel(u'W8', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1, weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)

    weights2, weights_array2 = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)
    dataMat, labelMat = loadDataSet()
    weights, weights_array = gradAscent(dataMat, labelMat)
    print("最终权重为：")
    print(weights_array[-1])
    testMat, testlebel = testDataSet()
    predict = sigmoid((np.mat(testMat) * np.mat(weights)))
    predict = predict.transpose()[0]
    print(map(int, predict))
    print(map(int, testlebel))

    print(predict)
    print(testlebel)
    count = 0
    for index in range(12):
        if int(predict.tolist()[0][index]) == testlebel[index]:
            count += 1
    print('11个测试组中正确数目为：%d,正确率为:%4.5f%%' % (count, count / 11 * 100))

