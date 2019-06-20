import numpy as np

"""
事务数据集
"""
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue               # 结点名字
        self.count = numOccur               # 计数值
        self.nodeLink = None                # 链接相似元素项
        self.parent = parentNode            # 指向父节点
        self.children = { }
    # 对count 变量增加给定值
    def inc(self, numOccur):
        self.count += numOccur
    # 文本形式输出 便于调试
    def disp(self, ind = 1):
        print(' '*ind, self.name, ' ', self.count)  #  ' ' *ind 控制格式输出 缩进
        for child in self.children.values():
            child.disp(ind+1)


"""
更新FP树
"""
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children: # 在树中的话就对相应节点+1
        inTree.children[items[0]].inc(count)
    else: # 不在树中的话，需要建子树
        inTree.children[items[0]] = treeNode(items[0], count, inTree)  # name number 以及父节点
        if headerTable[items[0]][1] == None:                        # 根节点为空的话 将新的子节点放到链表头里面
            headerTable[items[0]][1] = inTree.children[items[0]]    # 头指针表的作用：1 给定类型的第一个实例list的第二个元素 2 保存FP树种中每类元素的总数list的第一个元素
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:# [1::] 含义从[1:6]代表从下标1到6 左开右闭即 下标为 1 2 3 4 5 的元素 [1::] 即从1 到 :（所有）
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


"""
更新头节点
"""
def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest  = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


"""
创建FP-树
"""
def createTree(dataSet, minSup = 1):
    headerTable = {}
    # 第一遍遍历dataSet，构建全部的headerTable 相同元素地方相加，不同元素存在则置为1
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 对headertable进行最小支持度的筛选，将不满足支持度的删掉
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del (headerTable[k])
    # set 去重 这一步应该不存在多余元素，在第一遍遍历时候已经 将相同元素加到一起了
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0: return None,None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]         # 输出结果为{'test': [1, None], 'aaa': [2, None]} None 为 子树 value字段是list
    retTree = treeNode('Null Set', 1, None)
    # dataSet 是被处理过的， tranSet 是key字段  count 是value段
    # 第二遍遍历考虑 那些频繁元素
    for tranSet, count in dataSet.items():
        localD = {}
        # 根据全局频率对每个事务中的元素进行排序
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]   # headerTable[item] 等价于 item对应的  【number ,  None】
        if len(localD) > 0:
            # sorted 用法 reverse=true 降序排列 key=lambda表达式形式，即按照迭代对象的p[1]值（第二个数）进行排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)  # 使用排序后的频率项集对树进行填充
    return retTree, headerTable


def loadData(filename):
    dataSet = []
    with open(filename) as fr:
        for line in fr.readlines():
            dataSet.append(line.strip().split(','))
    return dataSet

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def ascendTree(leafNode, prefixPath):  # ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[0])]#(sort header table)
    for basePat in bigL:  #start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #print 'condPattBases :',basePat, condPattBases
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. mine cond. FP-tree
            print('conditional tree for: ',newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

if __name__ == '__main__':
    data = loadData('retail.data')
    print(np.shape(data))
    initSet = createInitSet(data)
    myFPtree, myHeaderTab = createTree(initSet, 10000)
    myFPtree.disp()

    freqItems = []
    mineTree(myFPtree, myHeaderTab, 10000, set([]), freqItems)
    print(freqItems)



