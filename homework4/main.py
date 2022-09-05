import pandas as pd
import numpy as np
import torch
from torch import nn

FEATURE_NUMBER = 18
HOUR_PER_DAY = 24

def DataProcess(df):
    x_list, y_list = [], []
    array = np.array(df).astype(float)

    for i in range(0, array.shape[0], FEATURE_NUMBER):
        for j in range(HOUR_PER_DAY - 9):
            mat = array[i:i+18, j:j+9]
            label = array[i+9,j+9] # 用PM2.5作为标签
            x_list.append(mat)
            y_list.append(label)
    x = np.float32(np.array(x_list))
    y = np.float32(np.array(y_list))
    return x, y

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        y_pred = self.linear_relu_stack(x)
        y_pred = y_pred.squeeze()
        return y_pred

if __name__ == '__main__':
    df = pd.read_csv('data.csv', usecols=range(2,26))
    # 将RAINFALL的空数据用0进行填充
    df[df == 'NR'] = 0       
    x, y = DataProcess(df)
    x = x.reshape(x.shape[0], -1)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    # 划分训练集和测试集
    x_train = x[:3000]
    y_train = y[:3000]
    x_test = x[3000:]
    y_test = y[3000:]
    
    model =  NeuralNetwork(x.shape[1])

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    # train
    print('START TRAIN')
    for t in range(2000):
        
        y_pred = model(x_train)

        loss = criterion(y_pred, y_train)
        if (t+1) % 50 == 0:
            print(t+1, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # test
    y_pred_test = model(x_test)
    loss_test = criterion(y_pred_test, y_test)
    print('TEST LOSS:', loss_test.item())
    




