import os

import jieba
from public_tools.progress_bar import *
import gensim.corpora
import gensim

"""
这段代码用来对en-zh的数据集进行清洗和分割。
"""


def split_ZH(orgin_path, write_path):
    """
    对中文数据集进行处理，删除<title>等标签信息
    :param orgin_path: 原始文件路径
    :param write_path: 写入文件路径
    :return: None
    """
    splited_data = []
    print("Reading origin file and split...")
    # 原始数据集
    with open(orgin_path, encoding='utf8') as file:
        line = file.readline()
        counter = 0
        while (line != ""):
            # Read by line
            counter+=1
            # delete <label> line
            if line.startswith(" <"):
                line = file.readline()
                continue
            # delete the spce
            line = line.replace(" ", "")
            # split sentence
            splited_line = jieba.lcut(line)
            # add space between word
            splited_line = ' '.join(splited_line)

            splited_data.append(splited_line)
            line = file.readline()
            progress(counter,250000) # 进度条。

    # 存储分割的词
    print("Split Success, Writing to the {}".format(write_path))
    with open(write_path, encoding='utf8', mode='w') as file:
        file.writelines(splited_data)
        file.close()
    print("split success")

def filter_EN(origin_path, write_path):
    """
    对原始英文数据集进行处理，删除<tittle>等标签信息
    :param origin_path: 原始数据集路径
    :param write_path: 写入数据集路径
    :return: None
    """
    # 处理英文数据
    splited_data = []
    with open(origin_path, encoding="utf8") as f:
        line = f.readline()
        while(line!=""):
            if line.startswith("<"):
                line = f.readline()
                continue
            splited_data.append(line.lower())
            line = f.readline()

    with open(write_path, encoding='utf8', mode='w') as f:
        f.writelines(splited_data)
    print("Save english")

def generate_word2vec(path, save_model_path):
    #
    data_ZN = []
    with open(path, mode='r', encoding='utf8') as f :
        line = f.readline()
        while line!="":
            temp = line.split(" ")
            data_ZN.append(temp)
            line = f.readline()
    # print(data_ZN)
    print("training word2vec")
    # 计算得到word2vect向量模型
    model = gensim.models.Word2Vec(sentences=data_ZN, vector_size=100, window=6, min_count=5, workers=4)
    model.save(save_model_path)


def convert_word_to_index(path, write_dir_path="../database/en-zh/generate" ,save_prex=""):
    data = []
    with open(path, mode='r', encoding='utf8') as f:
        line = f.readline()
        while line != "":
            temp = line.split(" ")
            data.append(temp)
            line = f.readline()

    # 生成词典
    dictionary = gensim.corpora.Dictionary(data)
    dictionary.save(os.path.join(write_dir_path, save_prex+"_dict"))
    data_ZH_idx = [dictionary.doc2idx(text) for text in data]
    with open(os.path.join(write_dir_path,save_prex+"_index.txt"), mode='w') as f:
        for line in data_ZH_idx:
            line = ' '.join(str(x) for x in line)
            f.write(line+"\n")


if __name__ == '__main__':
    orgin_path = "../database/en-zh/train.tags.en-zh.zh"
    write_path = "../database/en-zh/splited_train.tags.zh.txt"
    # split_ZH(orgin_path, write_path)
    # generate_word2vec(write_path, save_model_path="../database/en-zh/generate/word2vec.model")
    convert_word_to_index(write_path, write_dir_path="../database/en-zh/generate", save_prex="zh")
    #
    origin_path = "../database/en-zh/train.tags.en-zh.en"
    write_path ="../database/en-zh/splited_train.tags.en.txt"
    filter_EN(origin_path, write_path)
    convert_word_to_index(write_path, write_dir_path="../database/en-zh/generate", save_prex="en")


