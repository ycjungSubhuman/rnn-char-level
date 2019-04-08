'''
Model definitions
'''
import torch
import torch.nn as nn

RNN_TYPE = nn.RNN
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYER = 1
RNN_DROPOUT = 0.0

class CityNamePredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.rnn1 = RNN_TYPE(
            input_size, RNN_HIDDEN_SIZE, 
            RNN_NUM_LAYER, dropout=RNN_DROPOUT)
        self.linear = nn.Linear(RNN_HIDDEN_SIZE, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, name):
        hidden_0 = torch.zeros(RNN_NUM_LAYER, 1, RNN_HIDDEN_SIZE).cuda()
        hidden = hidden_0
        out, hidden = self.rnn1(name, hidden)
        return self.softmax(self.linear(out[name.size()[0]-1]))

class CityNameLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.NLLLoss()

    def forward(self, label, pred_vector):
        return self.loss(pred_vector, label)
