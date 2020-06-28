import torch
from AQI_DataSet import *
import matplotlib.pyplot as plt
import os

# 设置参数
learning_rate = 0.01
betas = (0.9, 0.999)
alpha = 0.9


# # swish激活函数
# class Swish(torch.nn.Module):
#     def __init__(self):
#         super(Swish, self).__init__()
#
#     def forward(self, x):
#         x = x * torch.sigmoid(x)
#         return x

# # SGD 就是随机梯度下降
# SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
# # momentum 动量加速,在SGD函数里指定momentum的值即可
# Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
# Adagrad = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
# # RMSprop 指定参数alpha
# RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
# # Adam 参数betas=(0.9, 0.99)
# Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))


class Net(torch.nn.Module):  # 继承 torch 的 Module（固定）
    def __init__(self, n_feature, n_hidden, n_output):  # 定义层的信息，n_feature多少个输入, n_hidden每层神经元, n_output多少个输出
        super(Net, self).__init__()  # 继承 __init__ 功能（固定）
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 定义隐藏层，线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 定义输出层线性输出

    def forward(self,
                x_train_normalized):  # x是输入信息就是data，同时也是 Module 中的 forward 功能，定义神经网络前向传递的过程，把__init__中的层信息一个一个的组合起来
        # 正向传播输入值, 神经网络分析出输出值
        x_train_normalized = torch.tanh(self.hidden(x_train_normalized))  # 定义激励函数(隐藏层的线性值)
        x_train_normalized = self.predict(x_train_normalized)  # 输出层，输出值
        return x_train_normalized


# 训练
def train(model, epochs, x_train, y_train, path):
    # optimizer 是训练的工具
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)
    # 训练与绘制训练图像
    plt.ion()  # 画图
    plt.show()
    for i in range(epochs):
        model.train()
        # net训练数据x, 输出预测值
        prediction = model(x_train)
        # 计算两者的均方误差
        loss = loss_func(prediction, y_train)
        # 上一步的更新梯度留在net.parameters()中，清空上一步的残余更新参数值
        optimizer.zero_grad()
        # 误差反向传播, 计算参数更新值
        loss.backward()
        # 更新参数
        optimizer.step()

        # 每五步绘制一次
        if i % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.title("Adam")
            X = np.linspace(0, len(np.array(y_train)), len(np.array(y_train)))
            plt.plot(X, y_train, marker='.', label="origin data")
            plt.xticks([])
            plt.plot(X, prediction.detach().numpy(), 'r-', marker='.', label="train", lw=1)
            plt.xticks([])
            plt.text(0, 0.95, 'Loss=%.4f' % loss, fontdict={'size': 20, 'color': 'red'})
            plt.text(0, 0.9, 'times=%d' % (i + 5), fontdict={'size': 15, 'color': 'red'})
            plt.legend(loc="upper right")
            plt.pause(0.1)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epochs}
    torch.save(state, path)
    plt.savefig('D:/pycharm/数据/Adam.png')
    # plt.pause(0)


def main():
    # 获取AQI数据
    x_train, x_test, y_train, y_test = DataSet()
    x_train_normalized, y_train_normalized = normalized(x_train, y_train)  # 归一化
    x_test_normalized, y_test_normalized = normalized(x_test, y_test)  # 归一化
    # 转换为tensor
    # train
    x_train_normalized = torch.from_numpy(x_train_normalized).clone().detach().float()
    y_train_normalized = torch.from_numpy(y_train_normalized).unsqueeze(1).clone().detach().float()
    # test
    x_test_normalized = torch.from_numpy(x_test_normalized).clone().detach().float()
    y_test_normalized = torch.from_numpy(y_test_normalized).unsqueeze(1).clone().detach().float()
    # 初始化网络
    net = Net(n_feature=6, n_hidden=9, n_output=1)
    # 定义路径
    path_sgd = "D:/pycharm/数据/BPNN_sgd.pt"
    path_sgd_m = "D:/pycharm/数据/BPNN_sgd_m.pt"
    path_adagrad = "D:/pycharm/数据/BPNN_adagrad.pt"
    path_rmsprop = "D:/pycharm/数据/BPNN_rmsprop.pt"
    path_adam = "D:/pycharm/数据/BPNN_adam.pt"
    train(net, epochs=5000, x_train=x_train_normalized, y_train=y_train_normalized, path=path_adam)


if __name__ == "__main__":
    main()
