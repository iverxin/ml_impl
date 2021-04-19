import pandas as pd
import numpy as np

import math

#决策树
class TreeNode:
    def __init__(self,flag,category,subtreenum,axis,value):
        self.flag = flag                                 #标记是不是叶子节点
        self.category = category                         #类别
        self.subtreenum = subtreenum                     #子树数量
        self.axis = axis                                 #分类依据属性（对应列）
        self.value = value                               #属性值
        self.left = None
        self.right = None
    def dfs(root):
        if(root == None):
            return
        print("---")
        print(root.flag,root.category)
        print(root.axis,root.value)
        print("---")
        TreeNode.dfs(root.left)
        TreeNode.dfs(root.right)

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
        for data in datas[:,-1]:                              #最后一列为label
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
        bestFeature = -1                                #用来标记最佳划分的特征属性，初始化为-1（对应数据集的某一列）
        bestvalue = 0
        bestGain = 0                                          #用来标记最划分后最大信息增益值

        for i in range(featrues):                             #对每个属性都进行计算划分后结果
            currentFeatureArray = datas[:,i]
            cmin = np.min(currentFeatureArray)                #当前属性最小值与最大值
            cmax = np.max(currentFeatureArray)
            for j in range(1,4):                              #切分值的选择为长度的1/4 ; 2/4 ; 3/4处
                splitnum = cmin + 0.25*j*(cmax - cmin)
                mindatas = DecisionTree.datasSplit(datas,splitnum,i,0)  #小于切分值splitnum的集合
                maxdatas = DecisionTree.datasSplit(datas,splitnum,i,1)  #大于切分值splitnum的集合
                newEntropy = len(mindatas)/length * DecisionTree.calcEntropy(mindatas) + len(maxdatas)/length * DecisionTree.calcEntropy(maxdatas)
                if baseEntropy - newEntropy > bestGain:
                    bestGain = baseEntropy - newEntropy
                    bestFeature = i
                    bestvalue = splitnum
        return bestFeature,bestvalue

    def judgeIsPurity(datas):
        length = float(len(datas))
        map = {}
        for label in datas[:,-1]:
            if label not in map.keys():
                map[label] = 0
            map[label]+=1
        if len(map) == 0:
            return -1
        for key in map.keys():
            if map.get(key)/length >= 0.88:
                return key
        return -2
    def createTree(datas):
        if DecisionTree.judgeIsPurity(datas) == -1:
            return None
        if DecisionTree.judgeIsPurity(datas) == -2:
            bestFeature,bestvalue = DecisionTree.selecBestDatasSplit(datas)
            root = TreeNode(-1,-1,2,bestFeature,bestvalue)
            leftdatas = DecisionTree.datasSplit(datas,bestvalue,bestFeature,0)
            rightdatas = DecisionTree.datasSplit(datas,bestvalue,bestFeature,1)

            root.left = DecisionTree.createTree(leftdatas)
            root.right = DecisionTree.createTree(rightdatas)
        else:
            root = TreeNode(1,DecisionTree.judgeIsPurity(datas),2,-1,-1)
        return root

def evaluation(root,data):
    if(root.flag == 1):
        return root.category
    if data[root.axis] <= root.value:
        return evaluation(root.left,data)
    else:
        return evaluation(root.right,data)

dataX,dataY = DecisionTree.loadDataSet("../database/iris_data.csv")
datas = np.concatenate((dataX,dataY),axis=1)

np.random.seed(np.random.randint(1,10))

train_datas = []
test_datas = []

for data in datas:
    if np.random.uniform() > 0.8:
        test_datas.append(data)
    else:
        train_datas.append(data)
train_datas = np.array(train_datas)
test_datas = np.array(test_datas)
root = DecisionTree.createTree(datas)

count = 0
for data in test_datas:
    y_pre = evaluation(root,data)
    if y_pre == data[-1]:
        count+=1
print("测试集总数为：{}".format(len(test_datas)))
print("正确样本数为：{}".format(count))
print("准确率为：{}".format(count / float(len(test_datas))))