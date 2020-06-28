import random
from AQI_DataSet import *


# random.seed(2)


# 生成区间[a, b)内的随机数
def my_rand(a, b):
    return (b - a) * random.random() + a


# 生成大小 I*J 的矩阵，默认零矩阵
def makeMatrix(I, J):
    return np.zeros([I, J], float)


# tanh
def tanh(x):
    return np.tanh(x)


# 函数 sigmoid 的派生函数, 为了得到输出 (即：y)
def tanh_backward(y):
    return 1.0 - y ** 2


class NN:
    ''' 三层反向传播神经网络 '''

    def __init__(self, ni, nh, no):
        # 输入层、隐藏层、输出层的节点（数）
        self.ni = ni + 1  # 增加一个偏差节点
        self.nh = nh + 1  # 增加一个偏差节点
        self.no = no

        # 激活神经网络的所有节点（向量）
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # 建立权重（矩阵）
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        # 设为随机值
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = my_rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = my_rand(-2.0, 2.0)

        # 最后建立Momentum动量因子（矩阵）
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('与输入层节点数不符！')

        # 激活输入层
        for i in range(self.ni - 1):
            # self.ai[i] = tanh(inputs[i])
            self.ai[i] = inputs[i]

        # 激活隐藏层
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)

        # 激活输出层
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = tanh(sum)

        return self.ao[:]

    def backPropagate(self, targets, lr, M):
        ''' 反向传播 '''
        if len(targets) != self.no:
            raise ValueError('与输出层节点数不符！')

        # 计算输出层的误差
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = tanh_backward(self.ao[k]) * error

        # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = tanh_backward(self.ah[j]) * error

        # 更新输出层权重
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] - lr * change + M * self.co[j][k]
                self.co[j][k] = change
                # print(N*change, M*self.co[j][k])

        # 更新输入层权重
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] - lr * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # 计算误差
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def weights(self):
        print('输入层权重:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('输出层权重:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, x_train, y_train, iterations=20000, lr=0.005, M=0.1):
        # lr: 学习速率(learning rate)
        # M: 动量因子(momentum factor)
        error_list = []
        k = 0
        for i in range(iterations):
            error = 0.0
            for j in range(x_train.shape[0]):
                inputs = x_train[j]
                targets = np.array([y_train[j]])
                self.update(inputs)
                error = error + self.backPropagate(targets, lr, M)
            if i % 100 == 0:
                error_list.append(error)
                if k >= 1 and error_list[k - 1] - error_list[k] < 0.0001:
                    lr = 0.9 * lr
                k += 1
                print('误差: %-.5f' % error, "lr:", lr)

    def test(self, x_test_normalized, y_test, y_test_normalized):
        result = []
        for x in x_test_normalized:
            r = self.update(x)
            # print(x, "->", r)
            result.append(r[0])
        print("均方误差:", self.mean_squared_error(np.array(y_test_normalized - np.array(result))))
        # 反归一化
        result = inverse_normalized(np.array(result), y_test)
        deviation = []
        for i in range(result.shape[0]):
            deviation.append(y_test[i] - result[i])
            print(i, "->", "预测值:", result[i], "实际值:", y_test[i], "偏差:", y_test[i] - result[i], "相对误差:",
                  np.fabs((y_test[i] - result[i]) / y_test[i]))
        print("反归一化后的均方误差:", self.mean_squared_error(deviation))
        print("反归一化后的平均偏差:", np.mean(np.fabs(deviation)))

    # 计算均方误差
    def mean_squared_error(self, deviation):
        return np.sum(np.power(np.array(deviation), 2)) / len(deviation)


def main():
    # 导入数据
    X_train, X_test, Y_train, Y_test = DataSet_Random(3)  # 获取随机拆分后的数据
    X_train, Y_train = normalized(X_train, Y_train)  # 归一化
    X_test_normalized, Y_test_normalized = normalized(X_test, Y_test)  # 归一化
    # 创建一个神经网络：输入层有6个节点，隐藏层节点数为2~12之间，输出层有一个节点
    n = NN(6, 9, 1)  # 输入层/隐藏层/输出层
    # 用一些模式训练它
    n.train(X_train, Y_train)
    # 测试训练的成果（不要吃惊哦）
    n.test(X_test_normalized, Y_test, Y_test_normalized)
    # 看看训练好的权重（当然可以考虑把训练好的权重持久化）
    n.weights()


if __name__ == '__main__':
    main()
