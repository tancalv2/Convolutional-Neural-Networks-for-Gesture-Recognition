import torch.nn as nn
import torch.nn.functional as F
import torch as torch

'''
    Write a model for gesture classification.
'''

######

# 3.2 YOUR CODE HERE

# best model

class CNN(nn.Module):

    def __init__(self, input_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 158, 10, padding=2, stride=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(158, 312, 10, padding=2, stride=1)
        self.conv3 = nn.Conv1d(312, 520, 10, padding=2, stride=1)

        self.fc1 = nn.Linear(520*8, 208)
        self.fc2 = nn.Linear(208, 104)
        self.fc3 = nn.Linear(104, 26)
        # self.fc3_bn = nn.BatchNorm1d(26)

    def forward(self, x):
        try:
            x = self.conv1(x.type(torch.FloatTensor).cuda())
        except:
            x = self.conv1(x.type(torch.FloatTensor))
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 520*8)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        # batch normalize before returning
        # x = self.fc3_bn(x)

        return F.log_softmax(x)

######
