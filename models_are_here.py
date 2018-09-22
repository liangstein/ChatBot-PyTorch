import torch
from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F

class Attention_layer(nn.Module):
    def __init__(self,sequence_length):
        super(Attention_layer, self).__init__()
        self.input_size=sequence_length
        self.output_size=sequence_length
        self.dense=nn.Linear(sequence_length,sequence_length)
        self.softmax=nn.Softmax(dim=-1)
    def forward(self, input_tensor): #input_tensor [B, T, D]
        y=self.softmax(self.dense(input_tensor.permute(0,2,1))) # [B,D,T]
        y=y.permute(0,2,1) # [B T D]
        y=input_tensor*y
        return y


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size,embed_size,padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True,batch_first=False)
    def forward(self, input_tensor, attention, hidden=None):
        '''
        :param input_seqs:
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        embedded = self.embedding(input_tensor) # [B T D]
        input_to_lstm=attention(embedded)  # [B T D]
        outputs, hidden = self.lstm(input_to_lstm.permute(1,0,2), hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs,hidden # [T B D]

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocabulary, n_layers=1, dropout=0):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dense=nn.Linear(20,20)
        self.softmax=nn.Softmax(dim=-1)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=False,batch_first=False)
        self.dense_output=nn.Linear(hidden_size,vocabulary)
        #self.softmax=nn.Softmax(dim=-1)
    def forward(self, input_tensor, sequence_length=20, hidden=None):
        '''
        :param input_seqs:
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        y=input_tensor.permute(1,2,0) # [B H T]
        y=self.softmax(self.dense(y))  # [B H T]
        y=y.permute(2,0,1)   # [T B H]
        y=y*input_tensor
        outputs, _ = self.lstm(y, hidden)
        y=self.dense_output(outputs)
        return y.permute(1,0,2) # [B T D]
