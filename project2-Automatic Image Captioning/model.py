import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN,self).__init__()
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        
        #torch.nn.Embedding(num_embeddings, embedding_dim):
            #num_embeddings (python:int) – size of the dictionary of embeddings
            #embedding_dim (python:int) – the size of each embedding vector
        self.embedding_vector=nn.Embedding(vocab_size,embed_size)
        
        #torch.nn.LSTM(*args, **kwargs):
            #input_size – The number of expected features in the input x
            #hidden_size – The number of features in the hidden state h
            #num_layers – Number of recurrent layers.
            #If True, then the input and output tensors are provided as (batch, seq, feature). 
        self.lstm=nn.LSTM(input_size=embed_size ,hidden_size=hidden_size ,num_layers=num_layers ,batch_first=True)
        
        #torch.nn.Linear(in_features, out_features):
            #in_features – size of each input sample
            #out_features – size of each output sample
        self.linear=nn.Linear(hidden_size,vocab_size)
    
    def forward(self, features, captions):
        #features is a batch of image passed through the encoder,it has the shape[batch size , embed size]
        #captions is a batch of tokens which all of them have the same length,it has the size[batch size, caption length]
        #those captions are toknized"converted to the corresponding integers",we need to remove the last token of them all which corresopnd
        #to the <'end'> as it should be the output of the last LSTM.
        
        #casting the captions:
        captions=captions[:,:-1]                 #remove last token,shape[batch size, caption length-1]
        embedding_captions=self.embedding_vector(captions) #shape[batch size, caption length-1, embedding size]
        
        #casting the features so they have same dimnesion as captions
        features=features.unsqueeze(1)  #shape became [batch size, 1, embedding size]
        
        #The LSTM is defined such that, as it sequentially looks at inputs, it expects that each individual input in a sequence is 
        #of a consistent size and so we embed the feature vector and each word so that they are embed_size
        inputs=torch.cat((features, embedding_captions), 1) #the LSTM inputs
        
        # the first value returned by LSTM is all of the hidden states throughout
        #the sequence. the second is just the most recent hidden state
        lstm_output,lstm_hidden = self.lstm(inputs)  #shape[batch size,captionlength-1,hidden size]
        
        output=self.linear(lstm_output)  #shape [batch size,caption length, vocab size]
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #inputs has the size [x,y,embedding]
        tokens=[]
        for i in range(max_len):
            events,states=self.lstm(inputs,states) #the output has size [x,y,hidden size],states here goes in recursion manner
            events_squeezed=events.squeeze(1)     #the shape now [x,hidden size]
            outputs=self.linear(events_squeezed)  #the shape now [x,vocab size]
            value,index=outputs.max(1)  #find the maximum of the vocab size'largest score' and it's index is where the word in the vocab
            tokens.append(index.item()) #got the value from the tensor and put it in the tokens list
            inputs=self.embedding_vector(index) #the output at t is input at t+1,has the shape[x,embedding]
            inputs=inputs.unsqueeze(1)         #now the shape is [x,y,embedding] pass it at the the next loop 
        return tokens    
       
            
        