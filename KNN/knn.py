import numpy as np
import pandas as pd
import math

_NN = 10

class KNN:
    def __init__(self,datas):
        self.datas = datas
        self.k = _NN
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
    def select_k_point(self,_data):
        map = {}
        length = len(self.datas)
        for index in range(length):
            map[index] = self.calcDiatance(self.datas[index][0:-1],_data[0:-1])
        list = sorted(map.items(),key=lambda item:item[1])
        return [item[0] for item in list][0:self.k]
    def calcCategory(self,_data):
        list = self.select_k_point(_data)
        category_map = {}
        for item in list:
            if category_map.get(self.datas[item][-1]) == None:
                category_map[self.datas[item][-1]] = 0
            category_map[self.datas[item][-1]] += 1
        _list = sorted(category_map.items(), key=lambda item: item[1],reverse=True)
        return _list[0][0]
    def evaluating(self,datas):
        length = float(len(datas))
        count = 0
        for _data in datas:
            y_pre = self.calcCategory(_data)
            print("实际类别:{} 预测类别:{}".format(_data[-1],y_pre))
            if y_pre == _data[-1]:
                count+=1
        print(count/length)
def loadDataSet(dataspath):
    '''
    :param dataspath:读入数据的地址，csv格式文件
    :return: n列的 np数组，前n-1列为特征，最后一列为label
    '''
    data_set = pd.read_csv(dataspath)
    data_set_train = data_set.iloc[:, 0:4]
    data_set_label = data_set.iloc[:, 5:6]
    return np.concatenate((np.array(data_set_train), np.array(data_set_label)), axis=1)


def main():
    datas = loadDataSet("../database/iris_data.csv")
    knn_model = KNN(datas)
    knn_model.evaluating(datas[25:125])

















if __name__ == '__main__':
    main()