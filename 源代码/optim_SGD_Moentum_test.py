# 带冲量的随机梯度下降SGD
from AQI_DataSet import *


# 多元数据
def SGD_Momentum():
    # 训练集，每个样本有三个分量
    X_train, X_test, Y_train, Y_test = DataSet_Random(3)  # 获取随机拆分后的数据
    x, y = normalized(X_train, Y_train)  # 归一化

    # 初始化
    m, dim = x.shape
    theta = np.zeros(dim)  # 参数
    alpha = 0.01  # 学习率
    momentum = 0.1  # 冲量
    threshold = 0.0001  # 停止迭代的阈值
    iterations = 5000  # 迭代次数
    error = 0  # 初始错误为0
    gradient = 0  # 初始梯度为0
    i = 0
    # 迭代开始
    for i in range(iterations):
        j = i % m
        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T,
                                     (np.dot(x, theta) - y))
        # 迭代停止
        if abs(error) <= threshold:
            break

        gradient = momentum * gradient - alpha * (x[j] *
                                                  (np.dot(x[j], theta) - y[j]))
        theta += gradient

    print('SGD_Momentum:迭代次数：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)


if __name__ == '__main__':
    SGD_Momentum()
