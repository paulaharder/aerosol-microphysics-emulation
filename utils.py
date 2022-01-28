''' A collection of functions and classes for the neural network emulator'''
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import argparse
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from regression_utils_ci import get_inv_scaler_x, get_inv_scaler_y, standard_transform_y_tend_inv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
stats = pickle.load(open('./data/dataset0/noss_stats_dataset0.p','rb'))
full_stats = pickle.load(open('./data/dataset0/full_stats_dataset0.p','rb'))
tend_stats = pickle.load(open('./data/dataset0/tend_stats_dataset0.p','rb'))
t_mean = Variable(torch.Tensor(tend_stats['ytrain_mean']),requires_grad=True).to(device)
t_std = Variable(torch.Tensor(tend_stats['ytrain_std']),requires_grad=True).to(device)
tend_mean = Variable(torch.Tensor(tend_stats['y_log_eps_mean']),requires_grad=True).to(device)
tend_std = Variable(torch.Tensor(tend_stats['y_log_eps_std']),requires_grad=True).to(device)
full_mean = Variable(torch.Tensor(tend_stats['X_log_eps_mean']),requires_grad=True).to(device)
full_std = Variable(torch.Tensor(tend_stats['X_log_eps_std']),requires_grad=True).to(device)
x_max = Variable(torch.Tensor(tend_stats['xtrain_max']),requires_grad=True).to(device)
y_max = Variable(torch.Tensor(tend_stats['ytrain_max']),requires_grad=True).to(device)
y_min = Variable(torch.Tensor(tend_stats['ytrain_min']),requires_grad=True).to(device)
mu_x = Variable(torch.Tensor(full_stats['xtrain_mean'][13:17]),requires_grad=True).to(device)
mu_y = Variable(torch.Tensor(tend_stats['ytrain_mean'][5:9]),requires_grad=True).to(device)
si_x = Variable(torch.Tensor(full_stats['xtrain_std'][13:17]),requires_grad=True).to(device)
si_y = Variable(torch.Tensor(tend_stats['ytrain_std'][5:9]),requires_grad=True).to(device)
#######arguments

def add_nn_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='dataset0', help="choose a data set to use")
    parser.add_argument("--eval_test", default='val', help="validation of test case")
    parser.add_argument("--scale_in", default='z', help="choose scaling, log, norm, standard, minmax")
    parser.add_argument("--scale_out", default='z_tend', help="choose scaling, log,  standard, minmax, logstandard")
    parser.add_argument("--tendency", type=bool, default=True, help="whether to predict the tendency or not")
    parser.add_argument("--lr", default=0.001, help="learning rate")
    parser.add_argument("--nlayers", default=2, help="number of hidden layers", type=int)
    parser.add_argument("--width", default=128, help="width of hidden layers")
    parser.add_argument("--loss", default='mse')
    parser.add_argument("--optimizer", default='adam')
    parser.add_argument("--weight_decay", default=1e-9)
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--early_stop", default=True)
    parser.add_argument("--no_ss", default=True)
    parser.add_argument("--mass_factor", default=1e8)
    return parser.parse_args()


######mass calculation
def mass_log(y):
    x = torch.exp(y[:,:28]*tend_std[:28]+tend_mean[:28])-1e-8
    so4 = 1e-8*torch.mean(torch.abs(torch.sum(y[:,28:32]*x[:,:4], dim=1)))
    bc = 1e+3*torch.mean(torch.abs(torch.sum(y[:,32:36]*x[:,5:9], dim=1)))
    oc =1e+4* torch.mean(torch.abs(torch.sum(y[:,36:40]*x[:,9:13],dim=1)))
    du = 1e+5*torch.mean(torch.abs(torch.sum(y[:,40:44]*x[:,13:17], dim=1)))
    mass = so4+bc+oc+du
    return mass

def overall_z_mass(x,y):  
    so4 = mse(mass_so4(x[:,:5]),mass_so4(y[:,:5]))
    bc = mse(mass_bc(x[:,5:9]),mass_bc(y[:,5:9]))
    oc =mse(mass_oc(x[:,9:13]),mass_oc(y[:,9:13]))
    du = mse(mass_du(x[:,13:17]),mass_du(y[:,13:17]))
    mass = so4+bc+oc+ du    
    return mass

def mass_z(y):
    summ = torch.sum(y, dim=1)
    so4_mass = torch.abs(summ)
    return so4_mass

def mass_so4(y):
    y = y*t_std[:5]+t_mean[:5]
    summ = torch.sum(y, dim=1)
    so4_mass = torch.abs(summ)
    return 1e-7*so4_mass

def mass_bc(y):
    y = y*t_std[5:9]+t_mean[5:9]
    summ = torch.sum(y, dim=1)
    so4_mass = torch.abs(summ)
    return 2*1e+4*so4_mass

def mass_oc(y):
    y = y*t_std[9:13]+t_mean[9:13]
    summ = torch.sum(y, dim=1)
    so4_mass = torch.abs(summ)
    return 2*1e+3*so4_mass

def mass_du(y):
    y = y*t_std[13:17]+t_mean[13:17]
    summ = torch.sum(y, dim=1)
    so4_mass = torch.abs(summ)
    return 1e-1*so4_mass

########losses

def relu_all(x):
    so4_pos = relu_so4(x)
    bc_pos = relu_bc(x)
    oc_pos = relu_oc(x)
    du_pos = relu_du(x)#
    num_pos = relu_num(x)
    wat_pos = relu_wat(x)
    pos = so4_pos + bc_pos + oc_pos + du_pos +num_pos +wat_pos
    #print(so4_pos, bc_pos,oc_pos, du_pos ,num_pos ,wat_pos)
    return pos

def relu_so4(x):
    return 1e-11*torch.mean(F.relu(-(x[:,:5]*t_std[:5]+x[:,28:33]*f_std[8:13]+t_mean[:5]+f_mean[8:13]))**2)

def relu_bc(x):
    return 1e+6*torch.mean(F.relu(-(x[:,5:9]*t_std[5:9]+x[:,33:37]*f_std[13:17]+t_mean[5:9]+f_mean[13:17]))**2)

def relu_oc(x):
    return 1e+7*torch.mean(F.relu(-(x[:,9:13]*t_std[9:13]+x[:,37:41]*f_std[17:21]+t_mean[9:13]+f_mean[17:21]))**2)

def relu_du(x):
    return 1e+3*torch.mean(F.relu(-(x[:,13:17]*t_std[13:17]+x[:,41:45]*f_std[21:25]+t_mean[13:17]+f_mean[21:25]))**2)

def relu_num(x):
    return 1e-8*torch.mean(F.relu(-(x[:,17:24]*t_std[17:24]+x[:,45:52]*f_std[25:32]+t_mean[17:24]+f_mean[25:32]))**2)

def relu_wat(x):
    return 1e+0*torch.mean(F.relu(-(x[:,24:28]*t_std[24:28]+t_mean[24:28]))**2)


def mass_mse(x, y):
    x_mass = mass_z(x)
    y_mass = mass_z(y)
    return mse(x_mass,y_mass)


def get_loss_function(args):
    loss_type = args.loss
    if loss_type == 'mse':
        loss_function = nn.MSELoss()
    elif loss_type == 'mass_conservation':
        loss_function = mass_violation_ss
    elif loss_type == 'log_sqrt':
        loss_function = log_sqrt_loss
    elif loss_type == 'mass_log_mse':
        loss_function = mass_log_mse
    return loss_function

###scores




#######training
        
def train_model(model, train_data, test_data, criterion, optimizer, epochs, input_dim, output_dim, args):
    best = 1e20
    patience = 0
    print(torch.cuda.is_available())
    for i in range(epochs):
        model_step(model, train_data, criterion, optimizer, i, epochs, args)
        val_loss = get_val_loss(model, test_data, criterion, i, epochs, args)
        checkpoint(model, val_loss, best, input_dim, output_dim, args)
        if args.early_stop:
            is_stop, patience = check_for_early_stopping(val_loss, best, patience)
        best = np.minimum(best, val_loss)
        if is_stop:
            break
                           
def model_step(model, train_data, criterion, optimizer, epoch, epochs, args):
    running_loss = 0
    for x, y in train_data:
        x = x.to(device)
        optimizer.zero_grad()
        model.to(device)
        output = model(x)
        y = y.to(device)
        loss = criterion(output,y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss = running_loss / len(train_data)
    print(epoch, 'train_loss', loss)

def get_val_loss(model, test_data, criterion, epoch, epochs, args):
    running_val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_data:
            x = x.to(device)
            output = model(x)
            y = y.to(device)
            loss =criterion(output,y)
            running_val_loss += loss.item()
    loss = running_val_loss/len(test_data)
    print(epoch, 'val_loss', loss)
    model.train()
    return loss

def model_name(args):
    name = args.model_id
    return name
    
def checkpoint(model, val_loss, best, input_dim, output_dim, args):
    if val_loss < best:
        checkpoint = {'model': model, out_features=output_dim, width=args.width),'state_dict': model.state_dict()}
        name = model_name(args)
        torch.save(checkpoint, './models/'+name+'.pth')
        

def check_for_early_stopping(val_loss, best, patience):
    is_stop = False
    if val_loss < best:
        patience = 0
    else:
        patience+=1
    if patience ==5:
        is_stop = True  
    return is_stop, patience
        
def create_dataloader(x,y, args):
    dataset = TensorDataset(torch.Tensor(x),torch.Tensor(y))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

def create_test_dataloader(x,y, args):
    dataset = TensorDataset(torch.Tensor(x),torch.Tensor(y))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

def create_report(model, X_test, y_test, args):
    #predict
    pred = model(torch.Tensor(X_test).to(device))
    pred = pred.cpu().detach().numpy()
    #scale back
    scale = get_inv_scaler_y(args.scale_out)
    #if log scale, need signs
    pred = scale.transform(pred)
    y_test = scale.transform(y_test)
    
    r2 = r2_score(pred[:,:4], y_test[:,:4], multioutput='raw_values')
    
    
    name = model_name(args)
    
    scores_dict = {'trained_r2': trained_r2,
                   'tend_r2': r2}
    
    
