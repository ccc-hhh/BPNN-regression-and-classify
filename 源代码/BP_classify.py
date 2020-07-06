from DataSet_wine import *
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 数据
x_train, x_test, y_train, y_test = DataSet_Random(0)

y_data = torch.from_numpy(y_train).clone().detach().long()
x_data = torch.from_numpy(normalized(x_train)).clone().detach().float()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, data):
        data = torch.relu(self.hidden(data))
        data = self.out(data)
        return data


def accuracy(predict, label):
    i = 0
    for p, l in zip(predict, label):
        if p == l:
            i += 1
    return i / len(label)


# n_hidden=2~11
net = Net(n_feature=13, n_hidden=15, n_output=3)
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
loss_func = torch.nn.CrossEntropyLoss()

# 多维数据绘图_t-SNE(t分布随机邻域嵌入)
t_sne = TSNE(n_components=2)
X = normalized(t_sne.fit_transform(x_data.numpy()))
plt.ion()
plt.show()

for t in range(2000):
    out = net(x_data)
    # print(out)
    loss = loss_func(out, y_data)  # loss是定义为神经网络的输出与样本标签y的差别，故取softmax前的值
    # print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        # plt.cla()
        # torch.max既返回某个维度上的最大值，同时返回该最大值的索引值
        _, prediction = torch.max(out, 1)  # 在第1维度取最大值并返回索引值
        # print(prediction)
        pred_y = prediction.data.numpy().squeeze()
        # print(pred_y)
        target_y = y_data.data.numpy()

        plt.cla()
        plt.scatter(X[:, 0], X[:, 1], c=pred_y)
        plt.xticks([])
        plt.text(0, 0.97, 'accuracy=%.4f' % accuracy(pred_y, target_y), fontdict={'size': 15, 'color': 'red'})
        plt.text(0, 0.9, 'times=%d' % (t + 5), fontdict={'size': 15, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
