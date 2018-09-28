import torch
import pickle
from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from models_are_here import Attention_layer,EncoderRNN,DecoderRNN
# load words dictionary
with open("word_index_dict","rb") as f:
    word_index_dict=pickle.load(f)

with open("index_word_dict","rb") as f:
    index_word_dict=pickle.load(f)

maxlen_q,maxlen_a=19,19
# build the model now
encoder=EncoderRNN(len(word_index_dict)+1,1024,1024)#.cuda()
decoder=DecoderRNN(1024,1024,len(index_word_dict)+2)#.cuda()
attention=Attention_layer(maxlen_q+1)#.cuda()
encoder.eval()
decoder.eval()
attention.eval()
params_encoder,params_decoder,params_attention=\
    list(encoder.parameters()),list(decoder.parameters()),list(attention.parameters())

# load weights into model
with open("weights/encoder", "rb") as f:
    weights_encoder=pickle.load(f)

with open("weights/decoder", "rb") as f:
    weights_decoder=pickle.load(f)

with open("weights/attention", "rb") as f:
    weights_attention=pickle.load(f)

for i in range(len(params_encoder)):
    params_encoder[i].data=weights_encoder[i].data.cpu()

for i in range(len(params_decoder)):
    params_decoder[i].data=weights_decoder[i].data.cpu()

for i in range(len(params_attention)):
    params_attention[i].data=weights_attention[i].data.cpu()

#encoder.cuda();decoder.cuda();attention.cuda() # uncomment if you have cuda gpus
def chat(string):
    q_vec=np.zeros((1,maxlen_q+1))
    for i,ele in enumerate(string):
        q_vec[0,i]=word_index_dict[ele]
    input_tensor=torch.from_numpy(q_vec).type(torch.LongTensor)
    outputs,_=encoder(input_tensor,attention)
    encoded_tensor=outputs
    answer=decoder(encoded_tensor[:])
    answer=answer.contiguous().view(-1,len(word_index_dict)+2)
    y=nn.Softmax(dim=-1)(answer)
    y=y.data.numpy()
    reply=[]
    for ele in y:
        indice=np.argmax(ele)
        if indice != 1:
            reply.append(index_word_dict[indice-1])
        else:
            break
    return "".join(reply)
