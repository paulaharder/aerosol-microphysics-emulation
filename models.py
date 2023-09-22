import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.autograd.set_detect_anomaly(True)

######models
'''class Base(nn.Module):
    def __init__(self, in_features, out_features, width, depth, contraint):
        super(Base, self).__init__()        
        self.fc_in = nn.Linear(in_features=in_features, out_features=width)
        self.act_1 = nn.ReLU()
        self.layer_1 = nn.Linear(in_features=width, out_features=width)
        self.act_2 = nn.ReLU()
        self.layer_2 = nn.Linear(in_features=width, out_features=width)
        self.act_3 = nn.ReLU()    
        self.fc_out = nn.Linear(in_features=width, out_features=out_features)
        #self.act_4 = nn.ReLU()
    def forward(self, x):
        x = self.fc_in(x)
        x = self.act_1(x)
        x = self.layer_1(x)
        x = self.act_2(x)
        #x = self.layer_2(x)
        #x = self.act_3(x)
        x = self.fc_out(x)
        #x = self.act_4(x)
        return x'''

class AddMassConstraints(nn.Module): #for z scale
    def __init__(self,mu_y, si_y):
        super(AddMassConstraints, self).__init__()
        self.mu_y = mu_y
        self.si_y = si_y
    def forward(self, y, x):
        #y += -np.mean(y[...,:5],axis=-1)
        y[...,:5] = 1/self.si_y[:5]*(y[:,:5]-torch.mean(self.mu_y[:5])-torch.mean(y[:,:5],dim=1).unsqueeze(1))
        y[...,5:9] = 1/self.si_y[5:9]*(y[:,5:9]-torch.mean(self.mu_y[5:9])-torch.mean(y[:,5:9],dim=1).unsqueeze(1))
        y[...,9:13] = 1/self.si_y[9:13]*(y[:,9:13]-torch.mean(self.mu_y[9:13])-torch.mean(y[:,9:13],dim=1).unsqueeze(1))
        y[...,13:17] = 1/self.si_y[13:17]*(y[:,13:17]-torch.mean(self.mu_y[13:17])-torch.mean(y[:,13:17],dim=1).unsqueeze(1))
        return y
    
class MultMassConstraints(nn.Module): #for n scale
    def __init__(self,mu_y, si_y):
        super(MultMassConstraints, self).__init__()
        self.mu_y = mu_y
        self.si_y = si_y
    def forward(self, y, x):
        out = y[:,:5]-torch.mean(self.mu_y[:5])
        #out_2 = out - torch.mean(y[:,:5],dim=1)
        y[...,:5] = 1/self.si_y[:5]*(y[:,:5]-torch.mean(self.mu_y[:5])-torch.mean(y[:,:5],dim=1).unsqueeze(1))
        y[...,5:9] = 1/self.si_y[5:9]*(y[:,5:9]-torch.mean(self.mu_y[5:9])-torch.mean(y[:,5:9],dim=1).unsqueeze(1))
        y[...,9:13] = 1/self.si_y[9:13]*(y[:,9:13]-torch.mean(self.mu_y[9:13])-torch.mean(y[:,9:13],dim=1).unsqueeze(1))
        y[...,13:17] = 1/self.si_y[13:17]*(y[:,13:17]-torch.mean(self.mu_y[13:17])-torch.mean(y[:,13:17],dim=1).unsqueeze(1))
        return y

class Base(nn.Module):
    def __init__(self, in_features, out_features, width, depth, constraint, mu_y=None, si_y=None, mu_x=None, si_x=None):
        super(Base, self).__init__()        
        self.fc_in = nn.Linear(in_features=in_features, out_features=width)
        self.hidden_layers = nn.ModuleList()
        for i in range(depth -1):
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Linear(in_features=width, out_features=width))
            self.hidden_layers.append(nn.ReLU())
        self.fc_out = nn.Linear(in_features=width, out_features=out_features)
        self.constraints = False
        if constraint=='add':
            self.constraints_layer = AddMassConstraints(mu_y,si_y)
            self.constraints  = True
        elif constraint=='compl':
            self.constraints_layer = CompletionLayer(mu_y,si_y)
            self.constraints  = True
        elif constraint=='randcompl':
            self.constraints_layer = RandCompletionLayer(mu_y,si_y)
            self.constraints  = True
        elif constraint=='corr_pre':
            self.constraints_layer = CorrectionLayerNew(mu_y,si_y)
            self.constraints  = True
        elif constraint=='corr':
            self.constraints_layer = CorrectionLayer(mu_y,si_y,mu_x,si_x)
            self.constraints  = True
    def forward(self, x_in):
        x = self.fc_in(x_in)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc_out(x)
        if self.constraints:
            x = self.constraints_layer(x,x_in)
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
class PositivityNN(nn.Module):
    def __init__(self, in_features, out_features, width):
        super(PositivityNN, self).__init__()        
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
        
    def forward(self,x,y):
        #x_out = torch.clone(x)
        x_out[:,4] =(- torch.sum(x[:,:4]*self.si_y[:4]+self.mu_y[:4], dim=1)-self.mu_y[4])/self.si_y[4]
        inds7 = [5,6,8]
        x_out[:,7] = (-torch.sum(x[:,inds7]*self.si_y[inds7]+self.mu_y[inds7], dim=1)-self.mu_y[7])/self.si_y[7]
        inds11 = [10,11,12]
        x_out[:,9] = (-torch.sum(x[:,inds11]*self.si_y[inds11]+self.mu_y[inds11], dim=1)-self.mu_y[11])/self.si_y[11]
        x_out[:,13] = (-torch.sum(x[:,14:17]*self.si_y[14:17]+self.mu_y[14:17], dim=1)-self.mu_y[13])/self.si_y[13] 
        return x_out
    
class RandCompletionLayer(nn.Module):
    def __init__(self, mu_y, si_y):
        super(RandCompletionLayer, self).__init__()
        self.mu_y = mu_y
        self.si_y = si_y
        
    def forward(self,x):
        #x_out = torch.clone(x)
        rand1 = np.randint(0,5)
        lis = [0,1,2,3,4]
        lis.remore(rand1)
        x_out[:,rand1] =(- torch.sum(x[:,lis]*self.si_y[lis]+self.mu_y[lis], dim=1)-self.mu_y[rand1])/self.si_y[rand1]
        inds7 = [5,6,7,8]
        rand2 = np.randint(5,9)
        inds7.remove(rand2)
        x_out[:,rand2] = (-torch.sum(x[:,inds7]*self.si_y[inds7]+self.mu_y[inds7], dim=1)-self.mu_y[rand2])/self.si_y[rand2]
        inds11 = [9,10,11,12]
        rand3 = np.randint(9,13)
        inds11.remove(rand3)
        x_out[:,rand3] = (-torch.sum(x[:,inds11]*self.si_y[inds11]+self.mu_y[inds11], dim=1)-self.mu_y[rand3])/self.si_y[rand3]
        inds = [13,14,15,16]
        rand4 = np.randint(13,17)
        inds.remove(rand4)
        x_out[:,rand4] = (-torch.sum(x[:,inds]*self.si_y[inds]+self.mu_y[inds], dim=1)-self.mu_y[rand4])/self.si_y[rand4] 
        return x_out

def neg_fraction(x,y, mu_y, si_y, mu_x, si_x):
    inds = [10]+[i for i in range(12,35)]
    y_orig = y[:,:28]*si_y[:28]+mu_y[:28] #output in original scal
    x_orig = x[:,inds]*si_x[inds]+mu_x[inds] #input in orginal scale
    pos = y_orig[:,:24]+x_orig
    return np.mean(pos.detach().cpu().numpy()<0,axis=0)

class CorrectionLayer(nn.Module):
    def __init__(self, mu_y, si_y, mu_x, si_x):
        super(CorrectionLayer, self).__init__()
        self.mu_y = mu_y
        self.si_y = si_y
        self.mu_x = mu_x
        self.si_x = si_x
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        
    def forward(self,y,x):
        inds = [10]+[i for i in range(12,35)]
        y_orig = y[:,:28]*self.si_y[:28]+self.mu_y[:28] #output in original scal
        x_orig = x[:,inds]*self.si_x[inds]+self.mu_x[inds] #input in orginal scale
        pos = self.relu1(y_orig[:,:24]+x_orig)
        y1 = pos - x_orig 
        #print(np.mean((y1+x_orig).detach().cpu().numpy()<0,axis=0))
        #test past
        y_out = y.clone()
        y_out[:,:24] = (y1-self.mu_y[:24])/self.si_y[:24]      
        #y_out[:,24:28] = self.relu2(y_orig[:,24:28])
        y_out[:,24:28] = (self.relu2(y_orig[:,24:28])-self.mu_y[24:28])/self.si_y[24:28]
        print(np.mean((y_out[:,:24]*self.si_y[:24]+x[:,inds]*self.si_x[inds]+self.mu_x[inds]).detach().cpu().numpy()<0,axis=0))
        #print(neg_fraction(x,y_out, self.mu_y, self.si_y, self.mu_x, self.si_x))
        #test failed
        return y_out[:,:28]
    

class CorrectionLayerNew(nn.Module):
    def __init__(self, mu_x, si_x):
        super(CorrectionLayerNew, self).__init__()
        self.mu_x = mu_x
        self.si_x = si_x
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
    def forward(self,y,x):
        inds = [10]+[i for i in range(12,35)]
        pos = self.relu1(y[:,:24]+x[:,inds]+self.mu_x[:24]/self.si_x[:24])
        y[:,:24] = pos - x[:,inds] - self.mu_x[:24]/self.si_x[:24]     
        y[:,24:28] = self.relu2(y[:,24:28]+self.mu_x[24:28]/self.si_x[24:28])
        return y[:,:28]    
    
class CompletionNN(nn.Module):
    def __init__(self, in_features, out_features, width, mu_y, si_y, activate_completion):
        super(CompletionNN, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
        self.completion = CompletionLayer(mu_y, si_y)
        self.completion_active = activate_completion
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        if self.completion_active:
            x = self.completion(x)
        return x
    
class CorrectionNN(nn.Module):
    def __init__(self, in_features, out_features, width, mu_y, si_y, mu_x, si_x, activate_correction):
        super(CorrectionNN, self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=width)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=width, out_features=width)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=width, out_features=out_features)
        self.correction = CorrectionLayer(mu_y, si_y, mu_x, si_x)
        self.correction_active = False
    def forward(self, x_in):
        x = self.fc1(x_in)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        if self.correction_active:
            x = self.correction(torch.cat((x, x_in[:,8:]), dim=1) )
        return x

        
        

        