import os
import random
import numpy as np  
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        attention_score = self.att(g)
        
        return torch.softmax(attention_score, 1)

class AttBiLSTM(nn.Module):
    def __init__(self):
        super(AttBiLSTM, self).__init__()

        ### Embedding
        self.embedding = '' # TODO Embedding layer

        ### BiLSTM
        self.lstm = nn.LSTM(input_size='TODO', num_layers=1, batch_first=True, hidden_size='TODO', bidirectional=True)

        ### Attention mechanism: define attn to be a KnowledgeAttn
        self.attn = Attention('LSTM INPUT SIZE * 2', 'OUTPUT_SIZE')

        ### Dropout + FC
        self.fc = nn.Linear('ATTN_INPUT_SIZE', 'OUTPUT_SIZE')
        self.do = nn.Dropout(p=0.5)


    def forward(self, x):
        out = self.embedding(x)
        out = self.lstm(out)
        out, beta = self.attn(out[0])
        out = self.do(F.relu(self.fc(out)))
        return out, beta