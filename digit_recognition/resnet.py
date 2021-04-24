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
import data_loader
import torch
import torch.nn.modules as nn
import jindutiao
from torch.utils.tensorboard import SummaryWriter

LR = 0.001
Batch_Size = 128
Epoch = 30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Resnet_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.avgpool2d = nn.AvgPool2d(2, stride=2)

        #输入部分
        self.conv2d_1 = nn.Conv2d(1,6,kernel_size=5,padding=2)
        self.batchnorm2d = nn.BatchNorm2d(6)

        #中间残差块
        self.conv2d_2 = nn.Conv2d(6,6,kernel_size=3,padding=1)

        #输出部分
        self.conv2d_3 = nn.Conv2d(6, 6, 5)
        self.flatten = nn.Flatten()
        self.sig = nn.Sigmoid()
        self.linear_1 = nn.Linear(6*5*5, 64)
        self.linear_2 = nn.Linear(64, 10)

    def forward(self,input):
        x = self.conv2d_1(input)
        x = self.batchnorm2d(x)
        x = self.relu(x)
        x = self.avgpool2d(x)

        for i in range(4):
            x = self.res_forward(x)

        x = self.conv2d_3(x)
        x = self.avgpool2d(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.sig(x)
        x = self.linear_2(x)
        x = torch.softmax(x, dim=1)
        return x

    def res_forward(self,input):
        x = self.conv2d_2(input)
        x = self.batchnorm2d(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.batchnorm2d(x)
        x += input
        x = self.relu(x)
        return x
def train():
    print("-----------------------Training-----------------------\n")
    train_datas = data_loader.loadDataSet("../database/HandwrittenDatas/train-images.idx3-ubyte","../database/HandwrittenDatas/train-labels.idx1-ubyte", 60000, Batch_Size)


    cnn = Resnet_model().to(device)
    loss_fn = nn.loss.BCELoss()
    opt = torch.optim.Adam(cnn.parameters(), lr=LR)

    for epoch in range(Epoch):
        aver_loss = 0
        counter = 0
        jindutiao_index = 0
        print("---epoch:{}---".format(epoch))
        for x, y in train_datas:
            x = x.to(device)
            y = y.to(device)
            x = x[:,np.newaxis]
            counter += 1
            y_pred = cnn(x)
            loss = loss_fn(y_pred, y)
            aver_loss += loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            jindutiao.progress(jindutiao_index,len(train_datas.dataset)/128-1)
            jindutiao_index += 1
        print("\nloss:{}".format(aver_loss / counter))
    return cnn

def Test(model):
    print("------------------------Testing------------------------\n")
    test_datas = data_loader.loadDataSet("../database/HandwrittenDatas/t10k-images.idx3-ubyte","../database/HandwrittenDatas/t10k-labels.idx1-ubyte",10000,Batch_Size)
    count = 0
    length = float(len(test_datas.dataset))
    jindutiao_index = 0
    for x,y in test_datas:
        x = x[:, np.newaxis]
        y_pred = model(x)
        y_pred = y_pred.detach().numpy()
        y = y.detach().numpy()

        y = np.argmax(y,axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        for o1,o2 in zip(y,y_pred):
            if o1==o2:
                count += 1
        jindutiao.progress(jindutiao_index,len(test_datas.dataset)/128)
        jindutiao_index += 1
    print("\n此Linear模型准确率为：{}".format(count/length))

def main():
    # model = train()
    # torch.save(model, './model_save/resnet_model.pth')
    model = torch.load('./model_save/resnet_model.pth').to('cpu')
    Test(model)


if __name__ == '__main__':
    print("-------------------------resnet-------------------------\n")
    main()