import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import gensim.models.keyedvectors as word2vec
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import numpy as np

class WordEmbedd():
    """
    Given a sentence of T words, convert each word into a real value vector by looking that word up in a word embedding matrix.

    The matrix will be of size V = to fixed size vocab. Matrix is to be learned from the corpus

    Word embedding is of size D which is a hyper parameter. In our paper, they chose 200 for d

    Word embedding  (ei) = word embedding matrix * 1 hot vector for that word

    Result is a vector of word embeddings to feed into the BiLSTM

    """

    def __init__(self) -> None:
        self.model = word2vec.KeyedVectors.load_word2vec_format('/Users/paulgerlich/dev/school/cs598dl4h/data/pubmed-w2v.bin', binary=True, unicode_errors='ignore')

    def embed(self, sentence):
        embedding = []

        for word in sentence.split(' '):
            index = self.model.key_to_index.get(word)
            if not index:
                if word == '<c>':
                    embedding.append(torch.zeros(200))
                continue

            embedding.append(torch.FloatTensor(self.model.vectors[index]))

        return torch.stack(embedding)
    


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
    def __init__(self, input_size, output_size):
        super(AttBiLSTM, self).__init__()

        ### Embedding
        self.embedding_do = nn.Dropout(0.3)

        ### BiLSTM
        self.lstm = nn.LSTM(input_size=input_size, num_layers=1, batch_first=True, hidden_size=100, bidirectional=True)
        self.lstm_do = nn.Dropout(0.3)

        ### Attention mechanism: define attn to be a KnowledgeAttn
        self.attn = Attention(input_size * 2)

        ### Dropout + FC
        self.fc = nn.Linear(input_size * 2, output_size)
        self.fc_do = nn.Dropout(p=0.5)


    def forward(self, x):
        out = self.lstm(x)
        out, beta = self.attn(out[0])
        out = self.do(F.relu(self.fc(out)))
        return out, beta


class CustomDataset(Dataset):
    
    def __init__(self, filename, ):    
        file = open(filename, 'r')
        self.data =  [line.lower() if '<c>' in line else None for line in file] # Remove any bad data
        self.data = list(filter(lambda item: item, self.data))
        self.word_embedding = WordEmbedd()

        self.label_to_index = {
            'present': 0,
            'absent': 1.0,
            'hypothetical': 2.0,
            'possible': 3.0,
            'conditional': 4.0,
            'associated_with_someone_else': 5.0
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index].replace('\n', '').split('----')
        text = data[0]
        label = self.label_to_index[data[1]]

        embeddings = self.word_embedding.embed(text)
        
        # return text as long tensor, labels as float tensor;
        return embeddings, torch.FloatTensor([label])

def collate_fn(data):
    text, labels = zip(*data)
    
    max_len = max([len(sentence) for sentence in text])

    padded_text = []
    for sentence in text:
        if max_len > len(sentence):
            padding = torch.stack([torch.zeros(200) * 1 for i in range(len(sentence), max_len)])

            padded_sentence = torch.stack([*sentence, *padding])
        else:
            padded_sentence = sentence

        padded_text.append(padded_sentence)

    labels = torch.stack(labels)

    return torch.stack(padded_text), torch.FloatTensor(labels)

def load_data(train_dataset, val_dataset, collate_fnn):
    batch_size = 32
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fnn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fnn)
    
    return train_loader, val_loader

def train(model, train_loader, val_loader, n_epochs, optimizer, criterion):
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
        p, r, f, roc_auc = eval(model, val_loader)
        print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, f: {:.2f}, roc_auc: {:.2f}'.format(epoch+1, p, r, f, roc_auc))
    return round(roc_auc, 2)

def evaluate_model(m, val_loader):
    m.eval()
    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    m.eval()
    for x, y in val_loader:
        y_logit = m(x)
        y_hat = (y_logit > 0.5).int()
        y_score = torch.cat((y_score,  y_logit.detach().to('cpu')), dim=0)
        y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)
    
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    return p, r, f, roc_auc


def test():
    train_dataset_data = CustomDataset('/Users/paulgerlich/dev/school/cs598dl4h/data/bet_train.txt')
    test_dataset_data = CustomDataset('/Users/paulgerlich/dev/school/cs598dl4h/data/test_data.txt')

    split = int(len(train_dataset_data)*0.8)
    lengths = [split, len(train_dataset_data) - split]

    train_dataset, val_dataset = random_split(train_dataset_data, lengths)
    train_loader, val_loader = load_data(train_dataset, val_dataset, collate_fn)

    # Padded size of max words in a single sentence
    input_size = len(train_dataset_data[0][0])

    # Input size --> Padded word vector size. Output size --> 6 (number of assertions)
    model = AttBiLSTM(input_size, 6)

    # load the loss function
    criterion = nn.BCELoss()

    # load the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 5

    train(model, train_loader, val_loader, n_epochs, optimizer, criterion)

    test_dataset, _ = random_split(test_dataset_data, lengths)
    test_loader, _ = load_data(train_dataset, val_dataset, collate_fn)

    test_results = evaluate_model(model, test_loader)

test()
