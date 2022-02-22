""" Here Various Neural Network Models are defined
@author: Philipp
"""
import torch
import torch.nn as nn


class ConvolutionalNet(nn.Module):
    def __init__(self, ):
        super(ConvolutionalNet, self).__init__()
        self.name = "Convolutional Net"
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=10),
            nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(
            nn.Conv1d(64, 16, kernel_size=10),
            nn.ReLU(inplace=True)
        )

        self.outfc = nn.Sequential(
            nn.Linear(16 * 746, 746), nn.ReLU(inplace=True),
            nn.Linear(746, 120), nn.ReLU(inplace=True),
            nn.Linear(120, 6), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        logits = self.outfc(torch.flatten(x3, 1))
        return logits


class LSTM_Net(nn.Module):
    def __init__(self, ):
        super(LSTM_Net, self).__init__()
        self.name = "LSTM Net"
        self.lstm = nn.LSTM(768, 16, num_layers=1, bidirectional=True, batch_first=True)
        self.outfc = nn.Sequential(nn.Linear(16, 6), nn.ReLU(inplace=True), )

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        out = self.outfc(hidden[-1])
        return out


# Logistic Regression is basically a single linear layer
# https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
class Logistic_Regression(nn.Module):
    def __init__(self, ):
        super(Logistic_Regression, self).__init__()
        self.name = "Logistic Regression"

        self.linear = torch.nn.Linear(768, 6)

    def forward(self, x):
        out = torch.squeeze(self.linear(x))
        return out


class MLP(nn.Module):
    def __init__(self, ):
        super(MLP, self).__init__()
        self.name = "MLP"

        self.outfc = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(inplace=True),
            nn.Linear(768, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 6), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.squeeze(self.outfc(x))
        return out
