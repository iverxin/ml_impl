import numpy as np
import math
import pandas as pd


class SVM:

    def __init__(self, X, Y):
        # load the data
        self.X = X
        self.Y = Y
        self.example_num = self.X.shape[0]  # len(self.X)

        # init alpha and b
        self.alpha = np.zeros(self.example_num)
        self.b = 0
        self.w = 0
        # hyparamters
        self.MAX_ITER_NUM = 100
        # 惩罚参数，是个超参数。越大对误分类的惩罚越大。
        self.C = 10  # TODO the C need to init the some method
        self.tol = 1e-3  # 容忍度，越小说明对KKT条件越苛刻
        self.E = np.zeros(self.example_num)

    def rand_j(self, i, m):
        """
        generate a index j of alpha2 different with i
        :param i: the alpha i
        :param m:
        :return: j index
        """
        j = np.random.randint(0, m)
        while (j == i):
            j = np.random.randint(0, m)
        return j

    def fxi(self, x_):
        """
        f(x) function
        :param X: the datasets
        :param x_: the predict data
        :return: the distance.
        """
        # ans = self.alpha * self.Y * self.kernel(X, x_) + self.b
        # w =0
        # for i in range(self.example_num):
        #     w += self.alpha[i] * self.Y[i] * self.X[i]
        # self.w = w
        # ans = w.dot(x_.T) + self.b
        # return ans

        ans = 0
        for i in range(self.example_num):
            ans += self.alpha[i] * self.Y[i] * self.kernel(self.X[i], x_)
        ans += self.b
        return ans

    def predict(self, x_):
        # ans = self.w.dot(x_.T) + self.b
        ans = self.fxi(x_)
        return np.sign(ans)

    def kernel(self, X1, X2, kernel_type='default', **kwargs):
        # TODO 加入其他的kernel方法
        if kernel_type == "default" or kernel_type == "linear":
            return np.dot(X1, X2.T)  # linear SVM
        if kernel_type == "polynomial":
            return np.dot(X1, X2.T) ** (kwargs['d'])

    def smo_b(self):
        """
        smo算法，alpha1和alpha2为随机选择。
        :return:
        """
        iter_num = 0
        while (iter_num < self.MAX_ITER_NUM):
            iter_num += 1

            for i in range(self.example_num):

                # sample alpha1 and alpha2
                # this method is a simple sample.
                # TODO make a better sample method
                index1 = i
                index2 = self.rand_j(index1, self.example_num)
                alpha1 = self.alpha[index1]
                alpha2 = self.alpha[index2]

                # calculate E(x_i)
                E1 = self.fxi(self.X[index1]) - self.Y[index1]
                E2 = self.fxi(self.X[index2]) - self.Y[index2]

                # calculate the boundary L and H
                y1 = self.Y[index1]
                y2 = self.Y[index2]
                if (y1 != y2):
                    L = max(0, alpha2 - alpha1)
                    H = min(self.C, self.C + alpha2 - alpha1)
                else:
                    L = max(0, alpha2 + alpha1 - self.C)
                    H = min(self.C, alpha2 + alpha1)

                # calculate the eta
                K11 = self.kernel(self.X[index1], self.X[index1])
                K21 = self.kernel(self.X[index2], self.X[index1])
                K22 = self.kernel(self.X[index2], self.X[index2])
                K12 = self.kernel(self.X[index1], self.X[index2])
                eta = K11 + K22 - 2. * K12

                # update the alpha2
                if eta == 0:
                    eta += 1e-6
                alpha2_unclip = alpha2 + y2 * (E1 - E2) / eta

                # clip alpha2
                alpha2_new = np.clip(alpha2_unclip, L, H)
                # calculate alpha1_new
                alpha1_new = alpha1 + y1 * y2 * (alpha2 - alpha2_new)

                # update to weights
                self.alpha[index1] = alpha1_new
                self.alpha[index2] = alpha2_new

                # update b
                b1 = -E1 - y1 * K11 * (alpha1_new - alpha1) - y2 * K21 * (alpha2_new - alpha2) + self.b
                b2 = -E2 - y2 * K12 * (alpha1_new - alpha1) - y2 * K22 * (alpha2_new - alpha2) + self.b
                self.b = (b1 + b2) / 2

    def smo_a(self):
        """
        基于启发式算法确定alpha1和alpha2
        :return:
        """
        iter_num = 0
        while iter_num < self.MAX_ITER_NUM:
            iter_num += 1
            for index1 in range(0, self.example_num):
                alpha1 = self.alpha[index1]
                self.E[index1] = self.fxi(self.X[index1]) - self.Y[index1]
                y1 = self.Y[index1]

                if (self.E[index1] * y1 > self.tol and alpha1 > self.tol) or (self.E[index1] * y1 < 0 and alpha1 < self.C):
                    # alpha1违反KKT条件。
                    # 开始寻找alpha2，选择E1-E2差值最大的。
                    delta = self.E[index1] - self.E
                    index2 = np.argmax(delta)
                    alpha2 = self.alpha[index2]
                    self.E[index2] = self.fxi(self.X[index2]) - self.Y[index2]

                    # calculate the boundary L and H
                    y1 = self.Y[index1]
                    y2 = self.Y[index2]
                    if (y1 != y2):
                        L = max(0, alpha2 - alpha1)
                        H = min(self.C, self.C + alpha2 - alpha1)
                    else:
                        L = max(0, alpha2 + alpha1 - self.C)
                        H = min(self.C, alpha2 + alpha1)

                    # calculate the eta
                    K11 = self.kernel(self.X[index1], self.X[index1])
                    K21 = self.kernel(self.X[index2], self.X[index1])
                    K22 = self.kernel(self.X[index2], self.X[index2])
                    K12 = self.kernel(self.X[index1], self.X[index2])
                    eta = K11 + K22 - 2. * K12

                    # update the alpha2
                    if eta == 0:
                        eta += 1e-6
                    alpha2_unclip = alpha2 + y2 * (self.E[index1] - self.E[index2]) / eta

                    # clip alpha2
                    alpha2_new = np.clip(alpha2_unclip, L, H)
                    # calculate alpha1_new
                    alpha1_new = alpha1 + y1 * y2 * (alpha2 - alpha2_new)

                    # update to weights
                    self.alpha[index1] = alpha1_new
                    self.alpha[index2] = alpha2_new

                    # update b
                    b1 = -self.E[index1] - y1 * K11 * (alpha1_new - alpha1) - y2 * K21 * (alpha2_new - alpha2) + self.b
                    b2 = -self.E[index2] - y2 * K12 * (alpha1_new - alpha1) - y2 * K22 * (alpha2_new - alpha2) + self.b
                    self.b = (b1 + b2) / 2


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
    data = loadDataSet("../database/iris_data.csv")
    data = data[0:100]  # 取前100examples
    X = data[:, 0:-1]  # 不取最后一列
    Y = data[:, -1]  # 去最后一列作为label
    example_num = data.shape[0]
    # format the labels to -1 and 1. the origin label is 0 and 1
    for i in range(example_num):
        if Y[i] == 0:
            Y[i] = -1
    svm_model = SVM(X, Y)
    svm_model.smo_a()


    y_pred = svm_model.predict(X)
    rights = (y_pred == Y).astype('float')
    rights = rights.sum() / rights.shape[0]
    print("准确率：{}%".format(rights*100))
    # test the kernel function


if __name__ == '__main__':
    main()
    print("Finished")
