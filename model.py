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
        self.embedds = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(hidden_size,vocab_size)
            
    def forward(self, features, captions):
        captions = captions[:,:-1]
        batch_size = features.size(0)
        embedd = self.embedds(captions)
        concat = torch.cat((features.unsqueeze(1),embedd),dim=1)
        lstm_out, _ = self.lstm(concat)
        lstm_out = self.linear(lstm_out)
        
        return lstm_out
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sentences = []
        for i in range(max_len):
            outputs, states = self.lstm(inputs,states)
            outputs = self.linear(outputs.squeeze(1))
            predicted = outputs.max(1)[1]
            sentences.append(predicted.item())
            inputs = self.embedds(predicted)
            inputs = inputs.unsqueeze(1)
        return sentences
        