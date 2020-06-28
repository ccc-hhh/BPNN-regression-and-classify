import numpy as np
from sklearn.model_selection import train_test_split


def DataSet():
    # 读数据
    AQI = np.genfromtxt("数据/成都AQI.csv", delimiter=",")
    # 取所有的行，和第一列之后的数据，因为第一列是标签，后面的是特征
    # PM2.5/PM10/SO2/CO/NO2/O3_8h
    X = AQI[:, 1:]
    X = X[1:, ]
    # 标签第一列
    y = AQI[:, 0]
    y = y[1:]
    # 默认1/5测试集，4/5训练集
    x_train = X[:130]
    x_test = X[130:]
    y_train = y[:130]
    y_test = y[130:]
    return x_train, x_test, y_train, y_test


def DataSet_Random(random_state):
    # 读数据
    AQI = np.genfromtxt("数据/成都AQI.csv", delimiter=",")
    X = AQI[:, 1:]
    X = X[1:, ]
    # 标签第一列
    y = AQI[:, 0]
    y = y[1:]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return x_train, x_test, y_train, y_test


# 全部的数据
def DataSet_All():
    # 读数据
    AQI = np.genfromtxt("数据/成都AQI.csv", delimiter=",")
    # 取所有的行，和第一列之后的数据，因为第一列是标签，后面的是特征
    # PM2.5/PM10/SO2/CO/NO2/O3_8h
    X = AQI[:, 1:]
    X = X[1:, ]
    # 标签第一列
    y = AQI[:, 0]
    y = y[1:]

    return X, y


# 对数据进行归一化处理
def normalized(x_data, y_data):
    e = 1e-7  # 防止出现0
    for i in range(x_data.shape[1]):
        max_num = np.max(x_data[:, i])
        min_num = np.min(x_data[:, i])
        x_data[:, i] = (x_data[:, i] - min_num + e) / (max_num - min_num + e)
    y_data = (y_data - np.min(y_data) + e) / (np.max(y_data) - np.min(y_data) + e)
    return x_data, y_data


# 对数据进行反归一化处理
def inverse_normalized(output, y_test):
    output = output * (np.max(y_test) - np.min(y_test)) + np.min(y_test)
    return output


# 设置数据形式
def form(data, targets):
    pat = []
    for i in range(data.shape[0]):
        pat.append([list(data[i]), [targets[i]]])
    return pat


if __name__ == "__main__":
    X, y = DataSet_All()
    X, y = normalized(X, y)
    print(X)
