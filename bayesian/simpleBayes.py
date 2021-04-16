import pandas as pd
import numpy as np
import math

class NaiveBayes:
    def __init__(self,dataspath):
        self.model = {}
        self.datas = self.loadDataSet(dataspath)
        self.categorys = set(self.datas[:,-1])
        self.features = len(self.datas[0])-1

    def loadDataSet(self,dataspath):
        '''
        :param dataspath:读入数据的地址，csv格式文件
        :return: n列的 np数组，前n-1列为特征，最后一列为label
        '''
        data_set = pd.read_csv(dataspath)
        data_set_train = data_set.iloc[:,0:4]
        data_set_label = data_set.iloc[:,5:6]
        return np.concatenate((np.array(data_set_train),np.array(data_set_label)),axis=1)

    def datasFilter(self,category):
        '''
        :param category: 抽取的类别
        :return: 此类别的所有数据
        '''
        retdatas = []
        for data in self.datas:
            if data[-1] == category:
                retdatas.append(data)
        return np.array(retdatas)

    def calcProbability(self,datas,axis):
        '''
        :param datas:某一类别全部数据
        :param axis: 当前计算的特征
        :return: 当前特征在当前类别下的 平均值 与 方差 ()元组形式
        '''
        items = datas[:,axis]
        average = np.average(items)
        variance = np.var(items)
        return  (np.around(average,2),np.around(variance,2))
    def generateModel(self):
        '''
        :return: 生成好的模型
        '''
        featuresnum = len(self.datas[0])-1
        categroynums = set(self.datas[:,-1])
        for categroy in categroynums:
            map = {}
            retdatas = self.datasFilter(categroy)
            for feature in range(featuresnum):
                (avg,var) = self.calcProbability(retdatas,feature)
                map[feature] = (avg,var)
            self.model[categroy] = map
        print("分类模型生成完成")
        return self.model

    def probability_density_function(self,value,avg,var):
        '''
        :param value:当前值
        :param avg: 平均数
        :param var: 方差
        :return: 当前值在高斯分布(avg,var)下的概率
        '''
        ratio = 1 / (math.sqrt(2 * math.pi * var))
        last = -math.pow((value - avg),2)/(2*var)
        return ratio * math.exp(last)

    def evaluation(self,testdatas):
        length = len(testdatas)
        correctcount = 0
        for data in testdatas:
            pre_categroy = -1
            max_pre = -1
            for categroy in self.categorys:
                temp_pre = 1
                for feature in range(self.features):
                    temp_pre = temp_pre * self.probability_density_function(data[feature],self.model.get(categroy).get(feature)[0],self.model.get(categroy).get(feature)[1])
                if(temp_pre > max_pre):
                    max_pre = temp_pre
                    pre_categroy = categroy
            if pre_categroy == data[-1]:
                correctcount += 1
        return correctcount/float(length)

model = NaiveBayes("../database/iris_data.csv")
model.generateModel()
pre = model.evaluation(model.datas)
print(pre)