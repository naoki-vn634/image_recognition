import torch
import torch.nn as nn
from torchvision import models

class CustomResnet18(nn.Module):

    def __init__(self,hidden_size = 256,out_size=2):

        super(CustomResnet18,self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.res = nn.Sequential(*list(resnet18.children())[:-1])

        self.fc1 = nn.Linear(512,hidden_size)
        self.dr1 = nn.Dropout()
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,out_size)

    def forward(self,x):
        h = self.res(x)
        h = h.view(len(h),-1)
        h = self.dr1(self.relu1(self.fc1(h)))
        y = self.fc2(h)
        
        return y

class CustomResnet18LSTM(nn.Module):
    
    def __init__(self,num_layers=1, input_size=20,hidden_size=64,output_size=2):
        
        super(CustomResnet18LSTM, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.res = torch.nn.Sequential(*list(resnet18.children())[:-1])
        self.fc1 = nn.Linear(512,hidden_size)
        self.dr1 = nn.Dropout()
        self.relu1 = nn.ReLU()

        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            )

        self.fc2 = nn.Linear(hidden_size,output_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self,x):

        batch_size,frames = x.shape[:2]
        x = x.view(-1,x.shape[2],x.shape[3],x.shape[4])
        h = self.res(x)
        h = self.dr1(self.relu1(self.fc1(h.view(-1,512))))
        h = h.view(batch_size,frames,-1)

        output, (h1,c1) = self.lstm(h, None)
        output = self.fc2(output[:,-1,:])

        return output

    def resnet_state(self,batch_size):
        self.h0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).to(self.device)
