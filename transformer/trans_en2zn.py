# -*- coding:utf-8 -*-
# !/usr/bin/python3
"""
Author: Spade
@Time : 2021/4/25 
@Email: spadeaiverxin@163.com
"""
from typing import Optional

import numpy as np
import torch

import torch.nn.modules as nn
import torchtext
from torch import Tensor
import os

from public_tools.logger import Logger


class MyTransformer(nn.Module):
    def __init__(self, dict_size_1:int ,dict_size_2:int , embedding_dim, nhead = 6,d_model = 512, input_len=50, output_len=50):
        super(MyTransformer, self).__init__()

        # self.pos_encoder = nn.PositionalEncoding()
        self.embeding1 = nn.Embedding(num_embeddings=dict_size_1,embedding_dim=embedding_dim) # num_embeedings: 字典大小， embeeding_dim:每个单词的输出维度
        self.embeding2 = nn.Embedding(num_embeddings=dict_size_2, embedding_dim=embedding_dim)
        # B L E

        # d_model就是attention层的embed_dim，指定每个输入词的维度
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=nhead, num_encoder_layers=6,
                                          num_decoder_layers=6,dim_feedforward=2048)
        # output: :math:`(T, N, E)`. target_len, batch_size, embedding_dim
        # 处理下outpu的输出

        self.linear1 = nn.Linear(embedding_dim , dict_size_2)
        # self.linear = nn.Linear(embedding_dim, dict_size_2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, src_mask = None, tgt_mask=None):
        """
        :param **kwargs:
        :param src: (B, S) : S是序列长度， B是batch_size
        :param tgt: 同src
        :return:
        """
        src = self.embeding1(src) # src (B, L, E) B batchsize, L 序列长度， E embedding dim
        tgt = self.embeding2(tgt)
        # tgt_mask = # tgt_mask部分mask掉输入前的所有东西。
        # 维度换， L,B,E
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask = tgt_mask) #
        # output::math: `(T, N, E)`.target_len, batch_size, embedding_dim

        output = output.permute(1,0,2) # (N, T, E)
        batch_size = output.shape[0]
        target_len = output.shape[1]

        output = output.reshape(batch_size*target_len, -1)
        output = self.linear1(output)
        output = output.reshape(batch_size, target_len, -1)
        output = self.softmax(output)
        return output # batchsize, targetLen, emmbedding_dim

ZH_TEXT = None
EN_TEXT = None

def create_dict(batch_size=64):
    tokenizer = lambda x: x.split()
    global ZH_TEXT
    global EN_TEXT
    ZH_TEXT = torchtext.legacy.data.Field(tokenize=tokenizer,
                                          init_token='<sos>', eos_token='<eos>', fix_length=100
                                          )
    EN_TEXT = torchtext.legacy.data.Field(tokenize=tokenizer,
                                          init_token='<sos>', eos_token='<eos>', fix_length=100
                                          )

    train, val = torchtext.legacy.data.TabularDataset.splits(path='../database/en-zh/generate/',
                                                             train='train_en_zh.csv',
                                                             validation='test_en_zh.csv',
                                                             skip_header=True, format='csv',
                                                             fields=[('en', EN_TEXT), ('zh', ZH_TEXT)])
    # 显示样本
    print(len(train[1].en), train[1].en)
    print(len(train[1].zh), train[1].zh)

    # 创建字典
    EN_TEXT.build_vocab(train)
    ZH_TEXT.build_vocab(train)
    print("en vocab {}".format(len(EN_TEXT.vocab)))
    print("zh vocab {}".format(len(ZH_TEXT.vocab)))

    print(EN_TEXT.vocab.itos[0])
    print(EN_TEXT.vocab.itos[1])
    print(EN_TEXT.vocab.itos[2])
    print(EN_TEXT.vocab.itos[3])
    print(EN_TEXT.vocab.itos[4])

    print(ZH_TEXT.vocab.itos[0])
    print(ZH_TEXT.vocab.itos[1])
    print(ZH_TEXT.vocab.itos[2])
    print(ZH_TEXT.vocab.itos[3])
    print(ZH_TEXT.vocab.itos[4])

    # 数据加载
    train_iter = torchtext.legacy.data.Iterator(train, sort_key=lambda x: len(x.en), batch_size=batch_size)
    val_iter = torchtext.legacy.data.Iterator(val, sort_key= lambda x:len(x.en), batch_size=batch_size)

    for batch in train_iter:
        print(batch.en[:, 0])  # seq_len, Batch size
        print(batch.en.shape, batch.zh.shape)
        break

    for batch in val_iter:
        print(batch.en[:, 0])  # seq_len, Batch size
        print(batch.en.shape, batch.zh.shape)
        a = batch.en[0][0]
        print(EN_TEXT.vocab.itos[int(a)])
        break

    return train_iter, val_iter

def create_padding_mask(seq, n_head=6, pad_idx=1): # b,S
    # 为1的为mask掉的
    seq_len = seq.shape[1]
    batch_size = seq.shape[0]
    mask = torch.eq(seq, torch.tensor(pad_idx)).float()
    mask = mask[:, np.newaxis ,np.newaxis, :]
    mask = mask.repeat(1,n_head,seq_len, 1) # batchsize
    mask = mask.reshape(batch_size*n_head, seq_len, seq_len)
    return mask# mask [batch_size, num_head, seq_Len, embeddding_dim


def create_behind_mask(batch_size, n_head, seq_len):
    # 为1的为mask掉的
    mask = torch.triu(torch.ones(seq_len, seq_len), 1)
    mask = mask* torch.ones(batch_size*n_head, 1, 1)
    return mask

def train(gpu=True):

    if not os.path.exists("model_save"):
        os.mkdir("model_save")
    MAX_STEPS = 2000

    batch_size = 16
    n_head = 5
    seq_len = 100

    train_iter, val_iter = create_dict(batch_size)
    en_vocb_size = len(EN_TEXT.vocab)
    zh_vocb_size = len(ZH_TEXT.vocab)
    print("EN-Vocab size:{}".format(en_vocb_size))
    print("ZH-Vocab size:{}".format(zh_vocb_size))

    model = MyTransformer(dict_size_1=en_vocb_size, dict_size_2=zh_vocb_size,
                          embedding_dim=100, nhead=n_head)
    if gpu:
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)

    logger = Logger("log.txt", print_flag=True)
    EPOCH = 1
    for epoch in range(EPOCH):
        loss_epoch = 0
        counter = 0

        for batch in train_iter:
            counter+=1
            # batch 的尺寸是 (seq_len, batch_size)
            # 这里进行一下维度调整。
            input = batch.en.permute(1,0) # batch_size, seq_len
            target = batch.zh.permute(1,0)
            # 准备mask
            input_mask = create_padding_mask(input,n_head=n_head)
            # print(input_mask.shape)
            # print(input_mask.shape)
            # print(input_mask)
            target_behind_mask = create_behind_mask(batch_size, n_head, seq_len)
            target_padding_mask = create_padding_mask(target, n_head=n_head)
            target_mask = torch.max(target_behind_mask, target_padding_mask)
            # print("inputMask \n",input_mask)
            # print("tgtMask \n",target_mask)

            if gpu:
                input = input.cuda()
                target = target.cuda()
                input_mask = input_mask.cuda()
                target_mask = target_mask.cuda()

            # 喂入模型
            output = model(input, target, src_mask =input_mask,tgt_mask = target_mask) # B,L,E
            target = target.reshape(batch_size*seq_len, -1).squeeze()
            output = output.reshape(batch_size * seq_len,-1)

            opt.zero_grad()
            loss = loss_fn(output, target)
            print(loss)
            loss.backward()
            # Update parameters
            opt.step()
            loss = loss.cpu()
            loss_epoch+=loss

        logger.log("epoch:{} | loss:{}".format(epoch, loss_epoch/counter))
        if epoch % 10 ==0 and epoch!=0:
            torch.save(model, "model_save/model_{}.model".format(epoch))
    torch.save(model, "model_save/model_final.model")

    print("save model to model_save")





if __name__ == '__main__':
    dict_size=10
    embedding_dim = 5
    seq_len = 3
    batch_size = 2
    # src = torch.randint(0,dict_size,[batch_size,seq_len-1])
    src = torch.tensor([[0,2,3],[1,0,3]])
    # src = torch.cat([src, torch.zeros(batch_size,1)], dim=-1)
    # tgt = torch.randint(0,dict_size,[batch_size,seq_len])
    # trans_model = MyTransformer(dict_size,dict_size, embedding_dim)
    # trans_model(src, tgt)
    # mask = create_padding_mask(seq_len)
    # print(mask)

    train()



