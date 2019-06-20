import numpy as np
import random
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    fr = open(fileName)
    stringArr = []
    # make = {'alfa-romero': 1, 'audi': 2, 'bmw': 3, 'chevrolet':4, 'dodge':5,
    #         'honda':6, 'isuzu':7, 'jaguar':8, 'mazda':9, 'mercedes-benz':10,
    #         'mercury':11, 'mitsubishi':12, 'nissan':13, 'peugot':14, 'plymouth':15,
    #         'porsche':16, 'renault':17, 'saab':18, 'subaru':19, 'toyota':20,
    #         'volkswagen':21, 'volvo':22, 'diesel':23, 'gas':24}
    # fuel_type = {'diesel':1, 'gas':2}
    # aspiration = {'std': 1, 'turbo': 2}
    # num_of_doors = {'four': 4, 'two': 2}
    # body_style = {'hardtop': 1, 'wagon': 2, 'sedan': 3, 'hatchback':4, 'convertible':5}
    # drive_wheels = {'4wd': 1, 'fwd': 2, 'rwd': 3}
    # engine_location = {'front': 1, 'rear': 2}
    # engine_type = {'dohc': 1, 'dohcv': 2, 'l': 3, 'ohc':4, 'ohcf':5, 'ohcv':6, 'rotor':7}
    # num_of_cylinders = {'eight':8, 'five':5, 'four':4, 'six':6, 'three':3, 'twelve':12, 'two':2}
    # fule_system = {'1bbl':1, '2bbl':2, '4bbl':3, 'idi':4, 'mfi':5, 'mpfi':6, 'spdi':7, 'spfi':8}
    for line in fr.readlines():                                     #逐行读取，滤除空格等
        lineArr = line.strip().split(',')
        if lineArr[1]=='?':
            lineArr[1] = (256-65)/2
        if lineArr[5]=='?':
            lineArr[5] = random.choice(['four','two'])
        if lineArr[18]=='?':
            lineArr[18] = (3.94-2.54)/2.0
        if lineArr[19]=='?':
            lineArr[19] = (4.17-2.07)/2.0
        if lineArr[21]=='?':
            lineArr[21] = (288-48)/2
        if lineArr[22]=='?':
            lineArr[22] = (6600-4150)/2
        if lineArr[25]=='?':
            lineArr[25] = (45400-5118)/2

        stringArr.append([float(lineArr[0]), float(lineArr[1]),
                          float(lineArr[9]),
                          float(lineArr[10]), float(lineArr[11]), float(lineArr[12]), float(lineArr[13]),
                          float(lineArr[16]),
                          float(lineArr[18]), float(lineArr[19]), float(lineArr[20]), float(lineArr[21]), float(lineArr[22]),
                          float(lineArr[23]), float(lineArr[24]), float(lineArr[25])])
    return stringArr

"""
参数：
    dataMat:用于进行PCA操作的数据集
    topNfeat：应用的N个特征
返回：
    降维后的数据集和被重构的原始数据（用于调试）
"""
def pca(dateMat, percentage=0.99):
    #计算均值，axis=0表示输出为行，求每一列平均
    meanVals = np.mean(dateMat, axis=0)
    #去均值
    meanRemoved = dateMat - meanVals
    #计算协方差矩阵， 寻找方差最大的方向a,Var(a'X)=a'Cov(X)a方向误差最大
    covMat = np.cov(meanRemoved, rowvar=0)
    #计算特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # print(eigVals)
    #对特征值进行从小到大排序
    eigValInd = np.argsort(eigVals)
    n = percentagen(eigVals, percentage)  #要达到percent的方差百分比，需要前n个特征向量
    #根据特征值排序的逆序得到N个最大的特征向量
    eigValInd = eigValInd[-1:-(n+1):-1]
    redEigVects = eigVects[:, eigValInd]
    #将数据转换到新的空间
    lowDDataMat = meanRemoved*redEigVects
    reconMat = (lowDDataMat*redEigVects.T) + meanVals
    print(lowDDataMat)
    return lowDDataMat, reconMat

def percentagen(eigVals,percentage):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    print(sortArray)
    for i in sortArray:
        tmpSum+=i
        num += 1
        print(arraySum * percentage)
        print(num)
        if tmpSum >= arraySum*percentage:
            return num


if __name__ == '__main__':
    dataMat = loadDataSet("imports-85.data")
    pca(dataMat)
