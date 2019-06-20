import numpy as np
from math import sqrt
from collections import Counter

def createDataSet():
    rax_data_x = [[3.393533211, 2.331273381],
                  [3.110073483, 1.781539638],
                  [1.343808831, 3.368360945],
                  [3.582294042, 4.679179110],
                  [2.280362439, 2.866990263],
                  [7.423436942, 4.696522875],
                  [5.745051997, 3.533989803],
                  [9.172168622, 2.511101045],
                  [7.792783481, 3.424088941],
                  [7.939820817, 0.791637231]
                  ]
    raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    x_train = np.array(rax_data_x)
    y_train = np.array(raw_data_y)
    return x_train, y_train

def KNN_classify(k, x_train, y_train, x):
    distances = [sqrt(np.sum((x_train-x)**2)) for x_train in x_train]
    nearest = np.argsort(distances)

    topk_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topk_y)

    return votes.most_common(1)[0][0]

if __name__ == '__main__':
	#创建数据集
	x_train, y_train = createDataSet()
	#测试集
	x = np.array([8.093607318, 3.365731514])
	#kNN分类
	result = KNN_classify(6, x_train, y_train, x)
	#打印分类结果
	print("predict:"+result)
