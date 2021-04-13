import pandas as pd
import numpy as np
import os
class DecisionTree:
    def loadDataSet(filepath):
        data_set = pd.read_csv(filepath)
        return data_set


data = DecisionTree.loadDataSet("../database/iris_data.csv")
print(data)