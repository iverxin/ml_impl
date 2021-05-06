# -*- coding:utf-8 -*-
# !/usr/bin/python3
"""
Author: Spade
@Time : 2021/4/30 
@Email: spadeaiverxin@163.com
"""


import os
class Logger(object):
    def __init__(self, path, over_write=True, print_flag = False):
        """
        Logger工具。
        :param path:log文件位置。
        :param over_write: 是否覆盖。
        """
        self.path = path
        if over_write and os.path.exists(path):
            os.remove(path)
            print("删除旧版本的{}".format(path))
        self.print_flag = print_flag

    def log(self, str):
        """
        记录log
        :param str:log内容。
        :return:
        """
        with open(self.path, mode='a', encoding="utf8") as f:
            f.write(str+"\n")
            f.flush()

        if self.print_flag:
            print(str)

if __name__ == '__main__':
    logger = Logger("test.log", print_flag=True)
    logger.log("dslfasdf")
    logger.log("dslfasdf")
    logger.log("dslfasdf")