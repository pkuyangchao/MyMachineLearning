# -*- coding: utf-8 -*-

from scipy import *
from math import log
import operator


"""
加载数据
"""


def createDataSet(filename):
    data = open(filename).readlines()
    data_set = []
    for line in data:
        format_line = line.strip().split()
        data_set.append(format_line)
    return data_set


"""
分割测试集和训练集
"""
def trainTestSplit(trainingSet, train_size):
    totalNum = int(len(trainingSet))
    trainIndex = list(range(totalNum))#存放训练集的下标
    x_test = []     #存放测试集输入
    x_train = []    #存放训练集输入
    trainNum = int(totalNum * train_size) #划分训练集的样本数
    for i in range(trainNum):
        randomIndex = int(random.uniform(0, len(trainIndex)))
        x_test.append(trainingSet[randomIndex])
        del trainIndex[randomIndex]#删除已经放入测试集的下标
    for i in range(totalNum-trainNum):
        x_train.append(trainingSet[trainIndex[i]])
    return x_train, x_test

"""
多数表决：返回标签列表中数量最大的类
"""


def majorityCnt(label_list):
    label_nums = {}
    for label in label_list:
        if label in label_nums.keys():
            label_nums[label] += 1
        else:
            label_nums[label] = 1
    sorted_label_nums = sorted(label_nums.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_nums[0][0]


"""
构建决策树
data_set：训练集
attribute_label：特征值列表
决策树用字典结构表示，递归的生成
"""


def createTree(data_set, attribute_label):
    label_list = [entry[-1] for entry in data_set]
    if label_list.count(label_list[0]) == len(label_list):  # 如果所有的数据都属于同一个类别，则返回该类别
        return label_list[0]
    if len(data_set[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(label_list)
    best_attribute_index = chooseBestFeatureToSplit(data_set)     # 选择最优特征
    best_attribute = attribute_label[best_attribute_index]   # 最优特征的标签
    decision_tree = {best_attribute: {}}       # 根据最优特征的标签生成树
    del (attribute_label[best_attribute_index])  # 找到最佳划分属性后需要将其从属性名列表中删除

    attribute_list = [entry[best_attribute_index] for entry in data_set]     # 得到训练集中所有最优特征的属性值
    attribute_set = set(attribute_list)     # 去掉重复的属性值
    for attribute in attribute_set:  # 属性的各个值
        sub_labels = attribute_label[:]     # 遍历特征，创建决策树
        decision_tree[best_attribute][attribute] = createTree(
            split_data_set(data_set, best_attribute_index, attribute), sub_labels)
    return decision_tree


"""
选择最优特征
"""


def chooseBestFeatureToSplit(data_set):
    num_attributes = len(data_set[0]) - 1  # 特征数量，减1是因为去掉了标签
    info_D = calcShannonEnt(data_set)  # 熵
    max_grian_rate = 0.0  # 最大信息增益
    best_attribute_index = -1   # 最优特征索引值

    for i in range(num_attributes):
        attribute_list = [entry[i] for entry in data_set]  # 获取data_set的第i个所有特征，此时为连续值
        info_A_D = 0.0  # 特征A对数据集D的信息增益
        split_info_D = 0.0  # 数据集D关于特征A的值的熵

        attribute_set = set(attribute_list)
        for attribute in attribute_set:  # 对每个属性进行遍历
            sub_data_set = split_data_set(data_set, i, attribute)
            prob = len(sub_data_set) / float(len(data_set))
            info_A_D += prob * calcShannonEnt(sub_data_set)
            split_info_D -= prob * log(prob, 2)
        if split_info_D == 0:
            split_info_D += 1
        grian_rate = (info_D - info_A_D) / split_info_D  # 计算信息增益比
        if grian_rate > max_grian_rate:
            max_grian_rate = grian_rate
            print(max_grian_rate)
            best_attribute_index = i
    return best_attribute_index


"""
计算数据集的熵
"""


def calcShannonEnt(data_set):
    num_entries = len(data_set)
    label_nums = {}  # 为每个类别建立字典，value为对应该类别的数目
    for entry in data_set:
        label = entry[-1]
        if label in label_nums.keys():
            label_nums[label] += 1
        else:
            label_nums[label] = 1
    info_D = 0.0
    for label in label_nums.keys():
        prob = float(label_nums[label]) / num_entries
        info_D -= prob * log(prob, 2)
    return info_D


"""
按按照给定特征划分数据集

data_set：待划分的数据集
index：划分属性的下标
value：划分属性的值

"""


def split_data_set(data_set, index, value):
    res_data_set = []   # 返回的数据集列表
    for entry in data_set:
        if entry[index] == value:  # 按数据集中第index列的值等于value的分数据集
            reduced_entry = entry[:index]
            reduced_entry.extend(entry[index + 1:])  # 划分后去除数据中第index列的值
            res_data_set.append(reduced_entry)
    return res_data_set     #返回划分后的数据集


"""
对一项测试数据进行预测，通过递归来预测该项数据的标签
decision_tree:字典结构的决策树
attribute_labels:数据的标签
one_test_data：预测的一项测试数据
"""


def classify(decision_tree, attribute_labels, one_test_data):
    first_key = list(decision_tree.keys())[0]    # 获取决策树结点
    second_dic = decision_tree[first_key]   # 下一个字典
    attribute_index = attribute_labels.index(first_key)
    res_label = None
    for key in second_dic.keys():
        if one_test_data[attribute_index] == key:
            if type(second_dic[key]).__name__ == 'dict':
                res_label = classify(second_dic[key], attribute_labels, one_test_data)
            else:
                res_label = second_dic[key]
    return res_label

if __name__ == '__main__':
    train_data = createDataSet("page-blocks.data")
    attribute_label = ['HEIGHT','LENGTH','AREA','ECCEN','P_BLACK','P_AND','MEAN_TR','BLACKPIX','BLACKAND','WB_TRANS']
    train_data,test_data = trainTestSplit(train_data, 0.1)
    decision_tree = createTree(train_data, attribute_label)

    # 递归会改变attribute_label的值，此处再传一次
    attribute_label = ['HEIGHT','LENGTH','AREA','ECCEN','P_BLACK','P_AND','MEAN_TR','BLACKPIX','BLACKAND','WB_TRANS']
    count = 0
    # 计算准确率
    for one_test_data in test_data:
        if classify(decision_tree, attribute_label, one_test_data) == one_test_data[-1]:
            count += 1
    accuracy = count / len(test_data)
    print('训练集大小%d，测试集大小%d，准确率为:%.1f%%' % (len(train_data), len(test_data), 100 * accuracy))
