import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import pandas as pd
from dataEngineer import read_trainData
train_data,train_labels = read_trainData()
train_data = torch.tensor(train_data.values)
train_labels = torch.tensor(train_labels.values)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc2(x)
        return x

net = Net()
print(net)
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
loss_fuction = torch.nn.MSELoss()
for t in range(100):
    prediction = net(train_data)
    loss = loss_fuction(prediction,train_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
