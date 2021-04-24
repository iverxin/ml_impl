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

LR = 0.0001
Batch_Size = 128
Epoch = 30

class Linear_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.Linear_1 = nn.Linear(28*28,128)
        self.sig = nn.Sigmoid()
        self.Linear_2 = nn.Linear(128, 84)
        self.Linear_3 = nn.Linear(84, 10)
    def forward(self,input):
        x = self.flatten(input)
        x = self.Linear_1(x)
        x = self.sig(x)
        x = self.Linear_2(x)
        x = self.sig(x)
        x = self.Linear_3(x)
        x = torch.softmax(x, dim=1)
        return x
def train():
    print("-----------------------Training-----------------------\n")
    train_datas = data_loader.loadDataSet("../database/HandwrittenDatas/train-images.idx3-ubyte","../database/HandwrittenDatas/train-labels.idx1-ubyte", 60000, Batch_Size)

    linear = Linear_model()
    loss_fn = nn.loss.BCELoss()
    opt = torch.optim.Adam(linear.parameters(), lr=LR)

    for epoch in range(Epoch):
        aver_loss = 0
        counter = 0
        jindutiao_index = 0
        print("---epoch:{}---".format(epoch))
        for x, y in train_datas:
            counter += 1
            y_pred = linear(x)
            loss = loss_fn(y_pred, y)
            aver_loss += loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            jindutiao.progress(jindutiao_index,len(train_datas.dataset)/128-1)
            jindutiao_index += 1
        print("\nloss:{}".format(aver_loss / counter))
    return linear
def Test(model):
    print("------------------------Testing------------------------\n")
    test_datas = data_loader.loadDataSet("../database/HandwrittenDatas/t10k-images.idx3-ubyte","../database/HandwrittenDatas/t10k-labels.idx1-ubyte",10000,128)
    count = 0
    length = float(len(test_datas.dataset))
    jindutiao_index = 0
    for x,y in test_datas:
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
    model = train()
    torch.save(model, './model_save/Linear_model.pth')
    Test(model)


if __name__ == '__main__':
    print("------------------------Linear------------------------\n")
    main()
