from AQI_DataSet import *


def RMSprop():
    # 训练集，每个样本有三个分量
    X_train, X_test, Y_train, Y_test = DataSet_Random(3)  # 获取随机拆分后的数据
    x, y = normalized(X_train, Y_train)  # 归一化

    # 初始化
    m, dim = x.shape
    theta = np.zeros(dim)  # 参数
    lr = 0.01  # 学习率
    threshold = 0.001  # 停止迭代的错误阈值
    iterations = 3000  # 迭代次数
    error = 0  # 初始错误为0
    decay_rate = 0.99  # 衰减率为0.99
    e = 1e-7  # 建议的默认值
    h = np.zeros(dim)
    i = 0
    for i in range(iterations):
        j = i % m
        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T,
                                     (np.dot(x, theta) - y))
        if abs(error) <= threshold:
            break

        gradient = x[j] * (np.dot(x[j], theta) - y[j])
        h *= decay_rate
        h += (1 - decay_rate) * gradient * gradient
        theta = theta - lr * gradient / (np.sqrt(h) + e)
        # print('迭代次数：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)
    print('RMSprop:迭代次数：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)


if __name__ == "__main__":
    RMSprop()
