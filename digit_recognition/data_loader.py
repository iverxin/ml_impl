'''
                            _ooOoo_
                           o8888888o
                           88" . "88
                           (| -_- |)
                           O\  =  /O
                        ____/`---'\____
                      .'  \\|     |//  `.
                     /  \\|||  :  |||//  \
                    /  _||||| -:- |||||-  \
                    |   | \\\  -  /// |   |
                    | \_|  ''\---/''  |   |
                    \  .-\__  `-`  ___/-. /
                  ___`. .'  /--.--\  `. . __
               ."" '<  `.___\_<|>_/___.'  >'"".
              | | :  `- \`.;`\ _ /`;.`/ - ` : | |
              \  \ `-.   \_ __\ /__ _/   .-` /  /
         ======`-.____`-.___\_____/___.-`____.-'======
                            `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      Buddha Bless, No Bug !
'''
import numpy as np
import struct
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def loadDataSet(images_path,labels_path,nums,batch):
    '''
    :param images_path: 特征集地址
    :param labels_path: 标签集地址
    :param nums: 取得的数据条数
    :param batch: 一个Batch_size大小
    :return: 划分好的数据集
    '''
    #图像读取
    f1 = open(images_path,'rb')
    image_buf = f1.read()
    image_index = 0
    image_index += struct.calcsize('>IIII')
    image_list = []
    for index in range(nums):
        temp = struct.unpack_from('>784B',image_buf,image_index)
        im = np.reshape(temp,(28,28))
        image_list.append(im)
        image_index += struct.calcsize('>784B')

    #label读取
    f2 = open(labels_path,'rb')
    label_buf = f2.read()
    label_index = 0
    label_index += struct.calcsize('>II')
    label_list = []
    for index in range(nums):
        temp = struct.unpack_from('>1B',label_buf,label_index)
        t = np.array([0,0,0,0,0,0,0,0,0,0])
        t[temp] = 1
        label_list.append(t)
        label_index += struct.calcsize('>1B')
    train_images = torch.tensor(np.array(image_list),dtype=torch.float32)
    train_label = torch.tensor(np.array(label_list),dtype=torch.float32)
    datas_td = TensorDataset(train_images,train_label)
    datas_dl = DataLoader(datas_td,batch_size=batch,shuffle=True)
    return datas_dl
