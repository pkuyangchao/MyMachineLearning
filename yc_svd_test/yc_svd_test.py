import numpy as np

def loadData(fileName):
    num = len(open(fileName).readline().split(","))-1
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split(",")
        for i in range(num):
            lineArr.append(curLine[i])
        dataMat.append(lineArr)

    return dataMat

def transform(dataMat):
    count = 0
    lent = len(dataMat)
    for i in range(30000):
        if dataMat[i][0] == "C":
            count += 1
        dataSet = [[0 for col in range(300)]for raw in range(count)]
        j = -1
    for i in range(30000):
        if dataMat[i][0] == "C":
            j += 1
        elif dataMat[i][0] == "V":
            index = int(dataMat[i][1])-1000
            dataSet[j][index] = 1
        else:
            continue

        return dataSet


def ecludSim(A,B):
    return 1.0/(1.0+np.linalg.norm(A-B))

def svdTest(dataMat, user, simMeas, item):
    dataMat = np.mat(dataMat)
    n = np.shape(dataMat)[0]
    print(n)
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,V = np.linalg.svd(dataMat)
    print(Sigma)
    Sig4 = np.mat(np.eye(4)*Sigma[:4])
    xformedItems = dataMat.T * U[:,:4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating==0 or j==item:continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print('the %d and %d similarity is:%f'%(item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity+userRating
    if simTotal == 0:return 0
    else:return ratSimTotal/simTotal,Sigma

def recommend(dataMat, user, N=3, simMeas=ecludSim, estMethod=svdTest):
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]  # find unrated items
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


if __name__ == '__main__':
    print(1)
    dataMat = loadData("anonymous-msweb.data")
    dataSet = transform(dataMat)
    similarity, sigma = svdTest(dataSet, 3, ecludSim, 3)
    # max = recommend(dataSet, 3, 3, ecludSim, svdTest)
    # print
    print(1)
    print(similarity)
    print(sigma)
