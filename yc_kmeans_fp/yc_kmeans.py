import numpy as np
import pandas as pd

"""
加载数据集
"""
def load_data(file_name):
    data = pd.read_csv(file_name)
    data = data.iloc[:,55:]  #直接使用归一化的数据
    return data

"""
随机选择质点
"""
def randCent(data,k,index):
    n = len(data[0])
    cent = np.zeros((k,1))
    max_v = max(np.mat(data)[:,index])
    min_v = min(np.mat(data)[:,index])
    mean = (max_v-min_v) / k
    for i in range(len(cent)):
        cent[i] = min_v+mean*i
    return cent

def distEclue(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


"""
K-means算法
"""
def kmeans(data,k,index,distMeas=distEclue,createCent=randCent):
    m = len(data)
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = createCent(data, k, index)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(np.mat(centroids)[j,:],np.mat(data)[i,index])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):#recalculate centroids
            ptsInClust = np.mat(data)[np.nonzero(clusterAssment[:,0].A==cent)[0]][:,index]#get all the point in this cluster
            #print(ptsInClust.shape)
            centroids[cent,:] = np.mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment

if __name__ == '__main__':
    data = load_data("Sales_Transactions_Dataset_Weekly.csv").values.tolist()
    for i in range(len(data[0])):
        centroids, clusterAssment = kmeans(data, 4, i, distEclue, randCent)
        print(centroids)
        print()