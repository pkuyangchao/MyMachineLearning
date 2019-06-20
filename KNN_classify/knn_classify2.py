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

class KNNClassifier:

    def __init__(self, k):
        '''初始化KNN分类器'''
        assert k >= 1, "k must be valid"
        self.k = k
        self._x_train = None
        self._y_train = None

    def fit(self, x_train, y_train):
        '''根据训练数据集训练KNN分类器'''

        # assert x_train.shape[0] == y_train.shape[0], \
        #     "the size of x_train must be equal to size of y_train"
        # assert self.k <= x_train.shape[0], \
        #     "the size of x_train must be at least k."

        self._x_train = x_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        '''给定预测数据集X_predict，返回表示X_predict的结果向量'''
        # assert self._x_train.shape is not None and self._y_train is not None, \
        #     "must fit before predict"
        # assert X_predict.shape[1] == self._x_train.shape[0], \
        #     "the feature number of X_predict must be equal to x_train."

        y_predict = [self._predict(x) for x in X_predict]

        return  np.array(y_predict)

    def _predict(self, x):
        '''给定单个待预测数据x，返回x的预测结果值'''
        # assert x.shape[1] == self._x_train.shape[0], \
        #     "the feature number of X_predict must be equal to x_train."

        distances = [sqrt(np.sum((x_train-x)**2))
                     for x_train in self._x_train]
        nearest = np.argsort(distances)

        topk_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topk_y)

        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k

if __name__ == '__main__':
	#创建数据集
    x_train, y_train = createDataSet()
    x = np.array([8.093607318, 3.365731514])
    knn_clf = KNNClassifier(k=6)
    knn_clf.fit(x_train, y_train)
    result = knn_clf.predict(x)
    print(result[0])

