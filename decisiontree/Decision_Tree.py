import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math

class DecisionTree:

    #数据读入，返回特征集与Label集
    def loadDataSet(filepath):
        data_set = pd.read_csv(filepath)
        data_set_train = data_set.iloc[:,0:4]
        data_set_label = data_set.iloc[:,5:6]
        return np.array(data_set_train),np.array(data_set_label)

    #计算datas集合的信息熵
    def calcEntropy(datas):
        length = len(datas)
        map = {}
        for data in datas[:,-1]:
            if data not in map.keys():
                map[data] = 0
            map[data] = map[data] + 1
        res = 0
        for key in map.keys():
            precent = float(map[key])/length
            res = res + precent * math.log(precent,2)
        return  -res

    def datasSplit(datas,value,axis,flag = 0):                #flag = 0取小于value的数据 flag = 1取大于value的数据
        retdatas = []
        for data in datas:
            if data[axis] <= value and flag == 0:
                retdatas.append(data)
            if data[axis] > value and flag == 1:
                retdatas.append(data)
        return np.array(retdatas)

    def selecBestDatasSplit(datas):
        length = float(len(datas))
        featrues = len(datas[0])-1                            #特征数量
        baseEntropy = DecisionTree.calcEntropy(datas)         #计算当前集合信息熵
        bestFeature_value = [-1,-1]                           #用来标记最佳划分的特征属性，初始化为-1（对应数据集的某一列）
        bestGain = 0                                          #用来标记最划分后最大信息增益值
        index = 0
        for i in range(featrues):                             #对每个属性都进行计算划分后结果
            currentFeatureArray = datas[:,i]
            cmin = np.min(currentFeatureArray)                #当前属性最小值与最大值
            cmax = np.max(currentFeatureArray)
            for j in range(1,4):                              #切分值的选择为长度的1/4 2/4 3/4处
                splitnum = cmin + 0.25*j*(cmax - cmin)
                mindatas = DecisionTree.datasSplit(datas,splitnum,i,0)  #小于切分值splitnum的集合
                maxdatas = DecisionTree.datasSplit(datas,splitnum,i,1)  #大于切分值splitnum的集合
                newEntropy = len(mindatas)/length * DecisionTree.calcEntropy(mindatas) + len(maxdatas)/length * DecisionTree.calcEntropy(maxdatas)
                if baseEntropy - newEntropy > bestGain:
                    bestGain = baseEntropy - newEntropy
                    bestFeature_value[0] = i
                    bestFeature_value[1] = splitnum
        return np.array(bestFeature_value)


dataX,dataY = DecisionTree.loadDataSet("../database/iris_data.csv")

datas = np.concatenate((dataX,dataY),axis=1)
