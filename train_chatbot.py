import torch
from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F
from models_are_here import Attention_layer,EncoderRNN,DecoderRNN
import pickle
import numpy as np
with open("less_length_questions","rb") as f:
    questions_tok=pickle.load(f)

with open("less_length_answers","rb") as f:
    answers_tok=pickle.load(f)

maxlen_q,maxlen_a=19,19
maxlength_list=[5,10,15,20]
'''with open("length_classified_questions","rb") as f:
    length_classified_questions=pickle.load(f)

with open("length_classified_answers","rb") as f:
    length_classified_answers=pickle.load(f)'''


with open("word_index_dict","rb") as f:
    word_index_dict=pickle.load(f)

with open("index_word_dict","rb") as f:
    index_word_dict=pickle.load(f)


setting_batch_size=200
encoder=EncoderRNN(len(word_index_dict)+1,1024,1024).cuda() # input has no EOS indice
decoder=DecoderRNN(1024,1024,len(index_word_dict)+2).cuda() # final output contains EOS indice
attention=Attention_layer(maxlen_q+1).cuda()
params_encoder,params_decoder,params_attention=\
    list(encoder.parameters()),list(decoder.parameters()),list(attention.parameters())
#attention_layer_list=[Attention_layer(ele).cuda() for ele in maxlength_list]
#attention_layers_params=[ele.parameters() for ele in attention_layer_list]
optimizer=optim.Adam(params_encoder+params_decoder+params_attention)
sheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=3)
loss=nn.CrossEntropyLoss(ignore_index=0)
steps_per_epoch=int(len(questions_tok)/setting_batch_size)
for epoch in range(2000):
    loss_lists = []
    all_labels=np.arange(0,len(questions_tok));np.random.shuffle(all_labels)
    batch_labels=np.array_split(all_labels,int(len(questions_tok)/setting_batch_size))
    for labels in batch_labels:
        batch_size=len(labels)
        q_vec=np.zeros((batch_size,maxlen_q+1))
        a_vec=np.zeros((batch_size,maxlen_a+1))
        for label_of_label,label in enumerate(labels):
            for j1,ele1 in enumerate(questions_tok[label]):
                q_vec[label_of_label,j1]=word_index_dict[ele1]
            for j2,ele2 in enumerate(answers_tok[label]):
                a_vec[label_of_label,j2]=word_index_dict[ele2]+1
            a_vec[label_of_label,j2+1]=1
        input_tensor=Variable(torch.from_numpy(q_vec).type(torch.LongTensor)).cuda()
        outputs,_=encoder(input_tensor,attention)
        encoded_tensor=outputs
        answer=decoder(encoded_tensor[:])
        # output has additional dimension due to EOS indice
        l=loss(answer.contiguous().view(-1,len(word_index_dict)+2),
               torch.from_numpy(a_vec).type(torch.LongTensor).view(-1).cuda())
        for i in params_attention:
            i.grad=None
        for i in params_encoder:
            i.grad=None
        for i in params_decoder:
            i.grad=None
        l.backward()
        optimizer.step()
        print(l)
        loss_lists.append(l.cpu().data.numpy())
    with open("losses","a") as f:
        epoch_loss=np.mean(loss_lists)
        f.write("Loss: {}\n".format(str(epoch_loss)))
    sheduler.step(epoch_loss)
    #check model weights
    with open("weights/encoder","wb") as f:
        pickle.dump([ele.cpu() for ele in params_encoder],f,protocol=pickle.HIGHEST_PROTOCOL)
    with open("weights/decoder","wb") as f:
        pickle.dump([ele.cpu() for ele in params_decoder],f,protocol=pickle.HIGHEST_PROTOCOL)
    with open("weights/attention", "wb") as f:
        pickle.dump([ele.cpu() for ele in params_attention], f, protocol=pickle.HIGHEST_PROTOCOL)