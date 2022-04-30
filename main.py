import os
import random
import numpy as np  
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class WordEmbedd():
    """
    Given a sentence of T words, convert each word into a real value vector by looking that word up in a word embedding matrix.

    The matrix will be of size V = to fixed size vocab. Matrix is to be learned from the corpus

    Word embedding is of size D which is a hyper parameter. In our paper, they chose 200 for d

    Word embedding  (ei) = word embedding matrix * 1 hot vector for that word

    Result is a vector one word embeddings to feed into the BiLSTM

    """


    def __init__(self) -> None:
        pass

class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        attention_score = self.att(g)
        
        return torch.softmax(attention_score, 1)

class AttBiLSTM(nn.Module):
    """
    
    Dropout layers should be placed after embedding layer, lstm layer, and final fc layer

    Referenced paper (Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification) that the BiLSTM was pulled from
    said that best values were 0.3, 0.3, 0.5 for DO respsectively
    
    """
    def __init__(self):
        super(AttBiLSTM, self).__init__()

        ### Embedding
        self.embedding = '' # TODO Embedding layer
        self.embedding_do = nn.Dropout(0.3)

        ### BiLSTM
        self.lstm = nn.LSTM(input_size='TODO', num_layers=1, batch_first=True, hidden_size='TODO', bidirectional=True)
        self.lstm_do = nn.Dropout(0.3)

        ### Attention mechanism: define attn to be a KnowledgeAttn
        self.attn = Attention('LSTM INPUT SIZE * 2', 'OUTPUT_SIZE')

        ### Dropout + FC
        self.fc = nn.Linear('ATTN_INPUT_SIZE', 'OUTPUT_SIZE')
        self.fc_do = nn.Dropout(p=0.5)


    def forward(self, x):
        out = self.embedding(x)
        out = self.lstm(out)
        out, beta = self.attn(out[0])
        out = self.do(F.relu(self.fc(out)))
        return out, beta