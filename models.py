import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

######models

class Base(nn.Module):
    def __init__(self, in_features, out_features, width):
        super(Base, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x
    
#base network concatenating the signs for log case
class SignExtBase(nn.Module):
    def __init__(self, in_features, out_features, width):
        super(SignExtBase, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
    def forward(self, x_in):
        x = self.fc1(x_in)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = torch.cat((x, x_in[:,32:]), dim=1)
        return x
    

class ClassificationNN(nn.Module):
    def __init__(self, in_features, out_features, width):
        super(ClassificationNN, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
        self.act3 = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        return x
    
#outputs inputs as well for positivity enforcement
class DoubleNN(nn.Module):
    def __init__(self, in_features, out_features, width):
        super(DoubleNN, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
    def forward(self, x_in):
        x = self.fc1(x_in)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x_out = torch.cat((x, x_in[:,8:]), dim=1) 
        return x_out
    
class CompletionLayer(nn.Module):
    def __init__(self, mu_y, si_y):
        super(CompletionLayer, self).__init__()
        self.mu_y = mu_y
        self.si_y = si_y
        
    def forward(self,x):
        x_out = torch.clone(x)
        x_out[:,4] =(- torch.sum(x[:,:4]*self.si_y[:4]+self.mu_y[:4], dim=1)-self.mu_y[4])/self.si_y[4]
        inds7 = [5,6,8]
        x_out[:,7] = (-torch.sum(x[:,inds7]*self.si_y[inds7]+self.mu_y[inds7], dim=1)-self.mu_y[7])/self.si_y[7]
        inds11 = [9,10,12]
        x_out[:,11] = (-torch.sum(x[:,inds11]*self.si_y[inds11]+self.mu_y[inds11], dim=1)-self.mu_y[11])/self.si_y[11]
        x_out[:,13] = (-torch.sum(x[:,14:17]*self.si_y[14:17]+self.mu_y[14:17], dim=1)-self.mu_y[13])/self.si_y[13] 
        return x_out

    
class CorrectionLayer(nn.Module):
    def __init__(self):
        super(CompletionLayer, self).__init__()
        
    def forward(self,x):
        neg = nn.ReLU(-(x[:,:28]*si_y+x[:,28:]*si_x+mu_y+mu_x[8:]))
        x_out = neg - x     
        return x_out
    
    
class CompletionNN(nn.Module):
    def __init__(self, in_features, out_features, width, mu_y, si_y):
        super(CompletionNN, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
        self.completion = CompletionLayer(mu_y, si_y)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x_out = self.completion(x)
        return x_out

        
        

        