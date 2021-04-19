import math
import numpy as np
import pandas as pd

#K均值 聚类

class K_means:
    def __init__(self,dataspath,k):
        '''
        :param dataspath:数据的地址
        :param k: 聚类簇数
        '''
        self.category = k
        self.model = {}
        self.datas = self.loadDataSet(dataspath)

    def loadDataSet(self,dataspath):
        '''
        :param dataspath:读入数据的地址，csv格式文件
        :return: np数组
        '''
        data_set = pd.read_csv(dataspath)
        return np.array(data_set.iloc[:,0:4])

    def calcDiatance(self,o1,o2):
        '''
        :param o1: 点o1
        :param o2: 点o2
        :return: o1与o2之间的相似度 ans越小表示类别越接近
        '''
        ans = 0
        for item1,item2 in zip(o1,o2):
            ans = ans + math.pow(item1-item2,2)
        return math.sqrt(ans)

    def iteration(self):
        avg_vector = {}
        #初始化k个向量
        for i in range(self.category):
            index = np.random.randint(0,len(self.datas))
            avg_vector[i] = self.datas[index]
        flag = 1
        while flag == 1:
            for cate in range(self.category):
                self.model[cate] = []
            for data in self.datas:
                best_cate = -1
                min_value = 1000
                for cate in range(self.category):
                    temp_value = self.calcDiatance(data,avg_vector[cate])
                    if temp_value < min_value:
                        min_value = temp_value
                        best_cate = cate
                self.model[best_cate].append(list(data))
            flag = -1
            for i in range(self.category):
                if self.calcDiatance(np.mean(self.model[i],axis=0),avg_vector[i]) != 0.0:
                    avg_vector[i] = np.mean(self.model[i],axis=0)
                    flag = 1

model = K_means("../database/iris_data.csv",3)
model.iteration()
print(model.model[0])
print(model.model[1])
print(model.model[2])