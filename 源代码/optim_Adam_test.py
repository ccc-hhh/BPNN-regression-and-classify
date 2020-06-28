# ADAM
from AQI_DataSet import *


def Adam():
    # 训练集，每个样本有三个分量
    X_train, X_test, Y_train, Y_test = DataSet_Random(3)  # 获取随机拆分后的数据
    x, y = normalized(X_train, Y_train)  # 归一化
    x, y = x[:10], y[:10]

    # 初始化
    r, dim = x.shape
    theta = np.zeros(dim)  # 参数
    lr = 0.01  # 学习率
    threshold = 0.001  # 停止迭代的错误阈值
    iterations = 3000  # 迭代次数
    error = 0  # 初始错误为0

    b1 = 0.9  # 建议的默认值
    b2 = 0.999  # 建议的默认值
    e = 1e-8  # 建议的默认值
    m = np.zeros(dim)
    v = np.zeros(dim)
    i = 0
    for i in range(iterations):
        j = i % r
        error = 1 / (2 * r) * np.dot((np.dot(x, theta) - y).T,
                                     (np.dot(x, theta) - y))
        if abs(error) <= threshold:
            break

        lr_t = lr * np.sqrt(1.0 - b2 ** (i + 1)) / (1.0 - b1 ** (i + 1))
        gradient = x[j] * (np.dot(x[j], theta) - y[j])
        m += (1.0 - b1) * (gradient - m)
        v += (1.0 - b2) * (gradient ** 2 - v)
        theta -= lr_t * m / np.sqrt(v + e)
        # print('迭代次数：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)
    print('Adam:迭代次数：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)


if __name__ == '__main__':
    Adam()
