'''
Model definitions
'''
import torch
import torch.nn as nn

RNN_TYPE = nn.RNN
RNN_HIDDEN_SIZE = 256
RNN_NUM_LAYER = 1
RNN_DROPOUT = 0.0

class CityNamePredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.rnn1 = RNN_TYPE(
            input_size, RNN_HIDDEN_SIZE, 
            RNN_NUM_LAYER, dropout=RNN_DROPOUT)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, name):
        hidden = self.rnn1.hidden_0()
        for i in range(name.size()[0]):
            out, hidden = self.rnn1(name[i], hidden)
        return torch.sigmoid(self.linear(out))

class CityNameLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.NLLLoss()

    def forward(self, label, pred_vector):
        return self.loss(pred_vector, label)
