# -*- coding:utf-8 -*-
#!/usr/bin/python3

import torch
from torch.nn import *

# feature_nums
FEATURE_NUMS = 10
SEQ_LEN = 5

# single Head
class Attention(Module):
    def __init__(self, feature_nums, seq_len):
        super().__init__()
        self.feature_nums = feature_nums
        self.seq_len = seq_len
        # 把需要训练的参数矩阵放到Parameter中，这样在调用model.parameters就有啦。
        self.W_q = Parameter(torch.randn((seq_len, feature_nums), dtype=torch.float))
        self.W_k = Parameter(torch.randn((seq_len, feature_nums), dtype=torch.float))
        self.W_v = Parameter(torch.randn((seq_len, feature_nums), dtype=torch.float))


    def forward(self, X):
        """
        See "Attention Is All You Need" for more details.

        Self Attention layer, 使用的是Dot-product版本。
        :param X: (seq, features)
        :return: 计算过attention的b。
        """

        # convert the dim
        # 转置
        X_T = X.permute(1,0) # 每个列代表一个example
        assert X_T.shape == (FEATURE_NUMS, SEQ_LEN)
        assert self.W_q.shape == (SEQ_LEN, FEATURE_NUMS)
        # calculate Q K V matrix
        Q = self.W_q.matmul(X_T) # 每个列代表一个q
        K = self.W_k.matmul(X_T)
        V = self.W_v.matmul(X_T)

        # calculate alpha 计算alpha
        alpha = K.permute(1,0) * Q # k需要转置成行向量, alpha矩阵列向量
        alpha = torch.softmax(alpha, dim=1) # alpha 进行softmax
        b = V * alpha
        return b


def main():
    X = torch.randn((SEQ_LEN, FEATURE_NUMS))
    attent = Attention(seq_len=SEQ_LEN, feature_nums=FEATURE_NUMS)
    print(attent(X))




if __name__ == '__main__':
    main()
