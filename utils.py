''' A collection of functions and classes for the neural network emulator'''

from models import Base, SignExtBase, ClassificationNN, PositivityNN, CompletionNN, CorrectionNN
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import csv
import os

EPS = 1e-20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#######arguments

def add_nn_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='train', help="train or eval with test")
    parser.add_argument("--signs", default=False, help="needed for log with mass reg.")
    parser.add_argument("--scale", default='z', help="z or log")
    parser.add_argument("--model", default="standard", help="standard, completion, correction, positivity, standard_log, log_mass, classification")
    parser.add_argument("--model_id", default="none")
    parser.add_argument("--log", default=False)
    parser.add_argument("--lr", default=0.001, help="learning rate")
    parser.add_argument("--width", default=128, type=int, help="width of hidden layers")
    parser.add_argument("--depth", default=2, type=int, help="number layers")
    parser.add_argument("--loss", default='mse')
    parser.add_argument("--optimizer", default='adam')
    parser.add_argument("--weight_decay", default=1e-9)
    parser.add_argument("--batch_size", default=2**12, type=int)
    parser.add_argument("--epochs", type=int,default=100)
    parser.add_argument("--early_stop", default=False)
    parser.add_argument("--save_val_scores", default=False)
    parser.add_argument("--old_data",default=False)
    parser.add_argument("--tend_full",default='tend')
    parser.add_argument("--alpha",default=0.99)
    parser.add_argument("--single",default='all')
    parser.add_argument("--constraint",default='none')
    parser.add_argument("--dataset",default='dataset14')
    return parser.parse_args()

def get_single_scores(true, pred, X_test, stats, args):
    #means = np.mean(true[:,:24]+X_val[:,8:], axis=0)
    N = true.shape[0]
    r2 = r2_score(true, pred, multioutput='raw_values')
    r2_log = r2_score(np.log(np.abs(true)+1e-8), np.log(np.abs(pred)+1e-8), multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error((true-stats['ytrain_mean'])/stats['ytrain_std'],( pred-stats['ytrain_mean'])/stats['ytrain_std'],multioutput='raw_values'))
    mae = np.sqrt(mean_absolute_error((true-stats['ytrain_mean'])/stats['ytrain_std'],( pred-stats['ytrain_mean'])/stats['ytrain_std'],multioutput='raw_values'))
    mean_bias = np.mean((true-pred)/stats['ytrain_std'], axis=0)
    '''mass_biases = np.mean(np.sum(pred, axis=1))
    masses_rmse =  np.sqrt(mean_squared_error(np.zeros((N,1)),np.sum(pred, axis=1)))
    full_pred = pred[:,:5]+X_val[:,8:13]
    neg_frac = np.mean(full_pred<0, axis=0)
    negative_mean = np.sum(full_pred[full_pred[:,0]<0,0])/(means[0]*N), np.sum(full_pred[full_pred[:,1]<0,1])/N,np.sum(full_pred[full_pred[:,2]<0,2])/N,np.sum(full_pred[full_pred[:,3]<0,3])/N,np.sum(full_pred[full_pred[:,4]<0,4])/N'''
    true_diff = np.copy(true)
    pred_diff = np.copy(pred)
    inds = [10]+[i for i in range(12,35)]
    #true_diff[:,1:24] -= X_val[:,12:]
    #true_diff[:,0] -= X_val[:,10] 
    #pred_diff[:,1:24]-= X_val[:,12:]
    #pred_diff[:,0] -= X_val[:,10] 
    true_diff[:,:24] += X_test[:,inds]#np.concatenate((X_val[:,16:18],X_val[:,19:24]),axis=1)
    pred_diff[:,:24] += X_test[:,inds]#np.concatenate((X_val[:,16:18],X_val[:,19:24]),axis=1)
    r2_diff = r2_score(true_diff, pred_diff, multioutput='raw_values')
    full_pred = np.zeros_like(pred)
    if args.tend_full == 'tend':
        #full_pred[:,0] = pred[:,0]+X_test[:,11]
        full_pred[:,:24] = pred[:,:24]+X_test[:,inds]
        full_pred[:,24:] = pred[:,24:]
        neg_frac = np.mean(full_pred<0,axis=0)
        #negative_mean = neg_mean(full_pred, stats)
    return  {'R2_single': r2, 'R2 log single': r2_log, 'RMSE single': rmse, 'mae single': mae, 'r2_full single':r2_diff, 'mean bias single': mean_bias, 'Negative fraction single': neg_frac}

def get_scores(true, pred, X_test, stats, args):
    r2=  r2_score(true, pred)
    r2_log =  r2_score(np.log(np.abs(true)+1e-8), np.log(np.abs(pred)+1e-8))
    rmse =  norm_rmse(true, pred, stats) #take from standardized values
    mass_biases = mass_middle(pred, stats) #individual masses divided by the respective abs mean
    masses_rmse = mass_rmse(true, pred, stats)#individual masses divided by the respective abs mean
    mae = np.sqrt(mean_absolute_error((true-stats['ytrain_mean'])/stats['ytrain_std'],( pred-stats['ytrain_mean'])/stats['ytrain_std'],multioutput='uniform_average'))
    full_pred = np.zeros_like(pred)
    inds = [10]+[i for i in range(12,35)]
    mean_bias = np.mean((true-pred)/stats['ytrain_std'])
    if args.tend_full == 'tend':
        full_pred[:,:24] = pred[:,:24]+X_test[:,inds]
        full_pred[:,24:] = pred[:,24:]
        neg_frac = np.mean(full_pred<0)
        negative_mean = neg_mean(full_pred, stats)
    return {'R2': r2, 'R2 log': r2_log, 'RMSE': rmse, 'mae': mae, 'mean bias': mean_bias, 'Mass Bias':mass_biases, 'Masses RMSE':masses_rmse,'mass rmse':np.mean(masses_rmse),'Negative fraction': neg_frac,'Negative mean' : negative_mean}

########
###stats
def calculate_stats(X_train, y_train, X_test, y_test, args):
    inds = [10]+[i for i in range(12,35)]
    means = np.mean(y_test[:,:24]+X_test[:,inds], axis=0)
    #means = np.mean(y_test[:,:24])
    X_log_eps = log_transform(X_train)
    y_log_eps = log_transform(y_train)
    
    so4 = np.concatenate((X_train[:,10],X_train[:,12],X_train[:,13],X_train[:,14],X_train[:,15]), axis=0)
    bc = np.concatenate((X_train[:,16],X_train[:,17],X_train[:,18],X_train[:,19]),axis=0)
    oc = np.concatenate((X_train[:,20],X_train[:,21],X_train[:,22],X_train[:,23]),axis=0)
    du = np.concatenate((X_train[:,24],X_train[:,25],X_train[:,26],X_train[:,27]),axis=0)
    
    

    so4_mean = np.mean(so4)  
    oc_mean = np.mean(oc)  
    du_mean = np.mean(du) 
    bc_mean = np.mean(bc)   
    so4_std = np.std(so4)  
    oc_std = np.std(oc)  
    du_std = np.std(du) 
    bc_std = np.std(bc)
    
    yso4_mean = np.mean(y_train[:,:5])  
    ybc_mean = np.mean(y_train[:,5:9])  
    yoc_mean = np.mean(y_train[:,9:13]) 
    ydu_mean = np.mean(y_train[:,13:17])   
    yso4_std = np.std(y_train[:,:5])  
    ybc_std = np.std(y_train[:,5:9])  
    yoc_std = np.std(y_train[:,9:13]) 
    ydu_std = np.std(y_train[:,13:17])
    
    y_species_mean = np.mean(y_train, axis=0)
    y_species_mean[:5] = yso4_mean 
    y_species_mean[5:9] = ybc_mean 
    y_species_mean[9:13] = yoc_mean 
    y_species_mean[13:17] = ydu_mean 
    y_species_std = np.std(y_train, axis=0)
    y_species_std[:5] = yso4_std 
    y_species_std[5:9] = ybc_std 
    y_species_std[9:13] = yoc_std 
    y_species_std[13:17] = ydu_std 
    
   
    
    y_species_min = np.min(y_train, axis=0)
    y_species_min[:5] = np.min(y_train[:,:5]) 
    y_species_min[5:9] = np.min(y_train[:,5:9])
    y_species_min[9:13] = np.min(y_train[:,9:13])
    y_species_min[13:17] = np.min(y_train[:,13:17])
    y_species_max = np.max(y_train, axis=0)
    y_species_max[:5] = np.max(y_train[:,:5]) 
    y_species_max[5:9] = np.max(y_train[:,5:9])
    y_species_max[9:13] = np.max(y_train[:,9:13])
    y_species_max[13:17] = np.max(y_train[:,13:17]) 
    
    inds = [10]+[i for i in range(12,35)]
    stats = {'xtrain_mean': np.mean(X_train, axis=0),
            'xtrain_std': np.std(X_train, axis=0),
            #'ytrain_mean': np.mean(np.concatenate((X_train[:,inds],y_train[:,24:]),axis=1), axis=0),
            #'ytrain_std': np.std(np.concatenate((X_train[:,inds],y_train[:,24:]),axis=1), axis=0),
             'xtrain_min': np.min(X_train, axis=0),
            'xtrain_max': np.max(X_train, axis=0),
             'ytrain_std': np.std(y_train, axis=0),
            'ytrain_mean': np.mean(y_train, axis=0),
            'ytrain_min': np.min(y_train, axis=0),
            'ytrain_max': np.max(y_train, axis=0),
             'yspecies_std': y_species_std,
            'yspecies_mean': y_species_mean,
             'yspecies_max': y_species_max,
            'yspecies_min': y_species_min,
             #'tend_mean': np.mean(y_train[:,:24]-X_train[:,inds], axis=0),
            #'tend_std': np.std(y_train[:,:24]-X_train[:,inds], axis=0),
             #'bc_mean':np.mean(y_train),
             #'bc_std':np.std(y_train),
            'so4_mean':so4_mean,
            'bc_mean':bc_mean,
            'oc_mean':oc_mean,
            'du_mean':du_mean,
             'yso4_mean':yso4_mean,
            'ybc_mean':ybc_mean,
            'yoc_mean':yoc_mean,
            'ydu_mean':ydu_mean,
             'yso4_std':yso4_std,
            'ybc_std':ybc_std,
            'yoc_std':yoc_std,
            'ydu_std':ydu_std,
            'X_log_eps_mean':np.mean(X_log_eps,axis=0),
            'X_log_eps_std': np.std(X_log_eps,axis=0),
            'y_log_eps_mean':np.mean(y_log_eps,axis=0),
            'y_log_eps_std': np.std(y_log_eps,axis=0),
            'means':means
            }
    

    global mu_y
    global si_y
    global mu_x
    global si_x
    inds = [10]+[i for i in range(12,35)]
    if args.scale == 'pre_z':
        stats['ytrain_mean'][:24]=stats['xtrain_mean'][inds]
        stats['ytrain_std'][:24]=stats['xtrain_std'][inds]
    
    mu_y = Variable(torch.Tensor(stats['ytrain_mean']),requires_grad=True).to(device)
    si_y = Variable(torch.Tensor(stats['ytrain_std']),requires_grad=True).to(device)
    mu_x = Variable(torch.Tensor(stats['xtrain_mean']),requires_grad=True).to(device)
    si_x = Variable(torch.Tensor(stats['xtrain_std']),requires_grad=True).to(device)
    return stats
    
####################################
######### TRANSFORMATION ############
def standard_transform_x(stats, x):
    return (x-stats['xtrain_mean'])/stats['xtrain_std']

def standard_transform_x_inv(stats, x):
    return x*stats['xtrain_std']+stats['xtrain_mean']

def standard_transform_y(stats, x):
    return (x-stats['ytrain_mean'])/stats['ytrain_std']

def standard_transform_y_inv(stats, x):
    return x[:,:28]*stats['ytrain_std']+stats['ytrain_mean']


def species_z_transform_y(stats, x):
    return (x-stats['yspecies_mean'])/stats['yspecies_std']

def species_z_transform_y_inv(stats, x):
    return x[:,:28]*stats['yspecies_std']+stats['yspecies_mean']

def species_n_transform_y(stats, x):
    return (x-stats['yspecies_min'])/(stats['yspecies_max']-stats['yspecies_min'])

def species_n_transform_y_inv(stats, x):
    return x[:,:28]*(stats['yspecies_max']-stats['yspecies_min'])+stats['yspecies_min']

def minmax_transform_x(stats, x):
    return (x-stats['xtrain_min'])/(stats['xtrain_max']-stats['xtrain_min'])

def minmax_transform_x_inv(stats, x):
    return x*(stats['xtrain_max']-stats['xtrain_min'])+stats['xtrain_min']

def minmax_transform_y(stats, x):
    return (x-stats['ytrain_min'])/(stats['ytrain_max']-stats['ytrain_min'])

def minmax_transform_y_inv(stats, x):
    return x*(stats['ytrain_max']-stats['ytrain_min'])+stats['ytrain_min']

def log_transform(x):
    return np.log(np.abs(x)+1e-8)

def exp_transform(x):
    return np.exp(x)-1e-8

def log_full_norm_transform_x(stats, x):
    x = log_transform(x)
    x = (x-stats['X_log_eps_mean'])/stats['X_log_eps_std']
    return x

def log_full_norm_transform_x_inv(stats, x):
    x = x*stats['X_log_eps_std']+stats['X_log_eps_mean']
    x = exp_transform(x)
    return x

def log_tend_norm_transform_y(stats, x):
    x = log_transform(x)
    x = (x-stats['y_log_eps_mean'])/stats['y_log_eps_std']
    return x

def log_tend_norm_transform_y_inv(stats, x):    
    x = x*stats['y_log_eps_std']+stats['y_log_eps_mean']
    x = exp_transform(x)
    return x

def bc_transform_x(stats, x):
    return (x-stats['bc_mean'])/stats['bc_std']

def bc_transform_x_inv(stats, x):
    return x*stats['bc_std']+stats['bc_mean']

#############
####data loaders
def create_dataloader(x,y, args):
    dataset = TensorDataset(torch.Tensor(x),torch.Tensor(y))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

def create_test_dataloader(x,y, args):
    dataset = TensorDataset(torch.Tensor(x),torch.Tensor(y))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

#########
########model criterion

def get_model(in_features, out_features,args, constraints_active):
    if args.model == 'standard' or args.model == 'standard_log':
        model = Base(in_features, out_features, args.width, args.depth, args.constraint, mu_y, si_y, mu_x, si_x)
    elif args.model == 'log_mass':
        model = SignExtBase(in_features, out_features, width=args.width)
    elif args.model == 'positivity':
        model = PositivityNN(in_features, out_features, width=args.width)
    elif args.model == 'classification':
        model = ClassificationNN(in_features=in_features, out_features=out_features, width=args.width)   
    elif args.model == 'completion':
        model = CompletionNN(in_features=in_features, out_features=out_features, width=args.width, mu_y=mu_y, si_y=si_y, activate_completion=constraints_active)
    elif args.model == 'correction':
        model = CorrectionNN(in_features=in_features, out_features=out_features, width=args.width, mu_y=mu_y, si_y=si_y, mu_x=mu_x, si_x=si_x,activate_correction=constraints_active)
    return model

def get_loss(output, y, args, x, stats):
    criterion = nn.MSELoss()
    crit2 = nn.MSELoss()
    class_criterion = nn.BCELoss()
    if args.loss == 'mse':
        return criterion(output, y)
    elif args.loss == 'tend_loss':
        inds = [10]+[i for i in range(12,35)]
        tend_output= output
        tend_output[:,:24]= output[:,:24]-x[:,inds]#(output[:,:24]*torch.Tensor(stats['ytrain_std'][:24]).to(device)+torch.Tensor(stats['ytrain_mean'][:24]).to(device))-(x[:,inds]*torch.Tensor(stats['ytrain_std'][:24]).to(device)+torch.Tensor(stats['ytrain_mean'][:24]).to(device))
        tend_y = y
        tend_y[:,:24]= y[:,:24]-x[:,inds]#(y[:,:24]*torch.Tensor(stats['ytrain_std'][:24]).to(device)+torch.Tensor(stats['ytrain_mean'][:24]).to(device))-(x[:,inds]*torch.Tensor(stats['ytrain_std'][:24]).to(device)+torch.Tensor(stats['ytrain_mean'][:24]).to(device))
        #tend_output = (tend_output-torch.Tensor(stats['tend_mean']).to(device))/torch.Tensor(stats['tend_std']).to(device)
        #tend_y = (tend_y-torch.Tensor(stats['tend_mean']).to(device))/torch.Tensor(stats['tend_std']).to(device)
        return (1-args.alpha)*criterion(output[:,:28], y[:,:28])+10e5*args.alpha*crit2(tend_output[:,:28], tend_y[:,:28])
    elif args.loss == 'tend_loss_bc':
        inds = [10]+[i for i in range(12,35)]
        tend_output= output-x[:,11:]#(output[:,:24]*torch.Tensor(stats['ytrain_std'][:24]).to(device)+torch.Tensor(stats['ytrain_mean'][:24]).to(device))-(x[:,inds]*torch.Tensor(stats['ytrain_std'][:24]).to(device)+torch.Tensor(stats['ytrain_mean'][:24]).to(device))
        tend_y= y-x[:,11:]#(y[:,:24]*torch.Tensor(stats['ytrain_std'][:24]).to(device)+torch.Tensor(stats['ytrain_mean'][:24]).to(device))-(x[:,inds]*torch.Tensor(stats['ytrain_std'][:24]).to(device)+torch.Tensor(stats['ytrain_mean'][:24]).to(device))
        #tend_output = (tend_output-torch.Tensor(stats['tend_mean']).to(device))/torch.Tensor(stats['tend_std']).to(device)
        #tend_y = (tend_y-torch.Tensor(stats['tend_mean']).to(device))/torch.Tensor(stats['tend_std']).to(device)
        return (1-args.alpha)*criterion(output[:,:28], y[:,:28])+args.alpha*crit2(tend_output, tend_y)
    elif args.loss == 'mse_mass':
        return args.alpha*criterion(output[:,:28], y[:,:28]), (1-args.alpha)*overall_z_mass(output, args.scale, stats)
    elif args.loss == 'mse_relu':
        return criterion(output[:,:28], y[:,:28])+relu_all(output)
    elif args.loss == 'mse_log_mass':
        return criterion(output, y)+torch.mean(mass_log(output))
    elif args.model == 'classification':
        return class_criterion(output, y)
    


#####################
######mass calculation
def mass_log(y):
    x = torch.exp(y[:,:28]*tend_std[:28]+tend_mean[:28])-1e-8
    so4 = 1e-8*torch.mean(torch.abs(torch.sum(y[:,28:32]*x[:,:4], dim=1)))
    bc = 1e+3*torch.mean(torch.abs(torch.sum(y[:,32:36]*x[:,5:9], dim=1)))
    oc =1e+4* torch.mean(torch.abs(torch.sum(y[:,36:40]*x[:,9:13],dim=1)))
    du = 1e+5*torch.mean(torch.abs(torch.sum(y[:,40:44]*x[:,13:17], dim=1)))
    mass = so4+bc+oc+du
    return mass

def mass_z(y):
    summ = torch.sum(y, dim=1)
    so4mass = torch.abs(summ)
    return so4_mass

def mass_so4(y):
    y = y*si_y[:5]+mu_y[:5]
    summ = torch.sum(y, dim=1)
    so4_mass = torch.abs(summ)
    return 1e-7*so4_mass

def mass_bc(y):
    y = y*si_y[5:9]+mu_y[5:9]
    summ = torch.sum(y, dim=1)
    so4_mass = torch.abs(summ)
    return 2*1e+4*so4_mass

def mass_oc(y):
    y = y*si_y[9:13]+mu_y[9:13]
    summ = torch.sum(y, dim=1)
    so4_mass = torch.abs(summ)
    return 2*1e+3*so4_mass

def mass_du(y):
    y = y*si_y[13:17]+mu_y[13:17]
    summ = torch.sum(y, dim=1)
    so4_mass = torch.abs(summ)
    return 1e-1*so4_mass

def species_mass(y, stats):
    so4_mu = Variable(torch.Tensor(stats['yso4_mean']),requires_grad=True).to(device)
    so4_sig = Variable(torch.Tensor(stats['yso4_std']),requires_grad=True).to(device)
    bc_mu = Variable(torch.Tensor(stats['ybc_mean']),requires_grad=True).to(device)
    bc_sig = Variable(torch.Tensor(stats['ybc_std']),requires_grad=True).to(device)
    oc_mu = Variable(torch.Tensor(stats['yoc_mean']),requires_grad=True).to(device)
    oc_sig = Variable(torch.Tensor(stats['yoc_std']),requires_grad=True).to(device)
    du_mu = Variable(torch.Tensor(stats['ydu_mean']),requires_grad=True).to(device)
    du_sig = Variable(torch.Tensor(stats['ydu_std']),requires_grad=True).to(device)
    so4 = torch.mean(torch.sum(y[:,:5], dim=1)+5*so4_mu/so4_sig)
    bc = torch.mean(torch.sum(y[:,5:9], dim=1)+4*bc_mu/bc_sig)
    oc = torch.mean(torch.sum(y[:,9:13], dim=1)+4*oc_mu/oc_sig)
    du = torch.mean(torch.sum(y[:,13:17], dim=1)+4*du_mu/du_sig)
    return so4+bc+oc+ du

#############

##############
########losses

def overall_z_mass(y, scale, stats):  
    if scale == 'z':
        so4 = torch.mean(mass_so4(y[:,:5]))
    
        bc = torch.mean(mass_bc(y[:,5:9]))
        oc = torch.mean(mass_oc(y[:,9:13]))
        du = torch.mean(mass_du(y[:,13:17]))
        mass = so4+bc+oc+ du    
    elif scale == 'species_z':
        mass = species_mass(y, stats)
    return mass
    

def relu_all(x):
    so4_pos = relu_so4(x)
    bc_pos = relu_bc(x)
    oc_pos = relu_oc(x)
    du_pos = relu_du(x)#
    num_pos = relu_num(x)
    wat_pos = relu_wat(x)
    pos = so4_pos + bc_pos + oc_pos + du_pos +num_pos +wat_pos
    return pos

def relu_so4(x):
    return 1e-11*torch.mean(F.relu(-(x[:,:5]*si_y[:5]+x[:,28:33]*si_x[8:13]+mu_y[:5]+mu_x[8:13]))**2)

def relu_bc(x):
    return 1e+6*torch.mean(F.relu(-(x[:,5:9]*si_y[5:9]+x[:,33:37]*si_x[13:17]+mu_y[5:9]+mu_x[13:17]))**2)

def relu_oc(x):
    return 1e+7*torch.mean(F.relu(-(x[:,9:13]*si_y[9:13]+x[:,37:41]*si_x[17:21]+mu_y[9:13]+mu_x[17:21]))**2)

def relu_du(x):
    return 1e+3*torch.mean(F.relu(-(x[:,13:17]*si_y[13:17]+x[:,41:45]*si_x[21:25]+mu_y[13:17]+mu_x[21:25]))**2)

def relu_num(x):
    return 1e-8*torch.mean(F.relu(-(x[:,17:24]*si_y[17:24]+x[:,45:52]*si_x[25:32]+mu_y[17:24]+mu_x[25:32]))**2)

def relu_wat(x):
    return 1e+0*torch.mean(F.relu(-(x[:,24:28]*si_y[24:28]+mu_y[24:28]))**2)


#######training
        
def train_model(model, train_data, test_data, optimizer, input_dim, output_dim, X_test, y_test, stats, args):
    best = 1e20
    patience = 0
    print('GPU available:', torch.cuda.is_available())
    is_stop = False
    if args.save_val_scores:
        val_r2 = []
        val_mse = []
        val_mass = []
        val_neg = []
    for i in range(args.epochs):
        model_step(model, train_data, optimizer, i, args, stats)
        val_loss = get_val_loss(model, test_data, i, args, stats)
        if args.save_val_scores:
            r2, mse, mass, neg = get_val_scores(model, X_test, y_test, i, stats, args)
            val_r2.append(r2)
            val_mse.append(mse)
            val_mass.append(mass)
            val_neg.append(neg)
        checkpoint(model, val_loss, best, input_dim, output_dim, args)
        if args.early_stop:
            is_stop, patience = check_for_early_stopping(val_loss, best, patience)
        best = np.minimum(best, val_loss)
        if is_stop:
            break
    if args.save_val_scores:
        save_validation_scores(val_r2, val_mse, val_mass, val_neg, args)
        
def neg_fraction(x,y):
    inds = [10]+[i for i in range(12,35)]
    y_orig = y[:,:28]*si_y[:28]+mu_y[:28] #output in original scal
    x_orig = x[:,inds]*si_x[inds]+mu_x[inds] #input in orginal scale
    pos = y_orig[:,:24]+x_orig
    return np.mean(pos.detach().cpu().numpy()<0,axis=0)

def neg_fraction_numpy(x,y,stats):
    inds = [10]+[i for i in range(12,35)]
    y_orig = y[:,:28]*stats['ytrain_std'][:28]+stats['ytrain_mean'][:28] #output in original scal
    x_orig = x[:,inds]*stats['xtrain_std'][inds]+stats['xtrain_mean'][inds] #input in orginal scale
    pos = y_orig[:,:24]+x_orig
    return np.mean(pos<0,axis=0)
                           
def model_step(model, train_data, optimizer, epoch, args, stats):
    running_loss = 0
    running_mass_loss = 0
    inds = [10]+[i for i in range(12,35)]
    for x, y in train_data:
        x = x.to(device)
        optimizer.zero_grad()
        model.to(device)
        output = model(x)
        
        #output = torch.sigmoid(output).clone()
        '''
        if epoch > 30:
            output = torch.sigmoid(output).clone()
        elif epoch > 20:
            output = F.relu(output)
        elif epoch > 10:
            output = F.leaky_relu(output)'''
        y = y.to(device)
        #print('neg', neg_fraction(x,output))
        if args.loss == 'mse_mass':
            mse_loss, mass_loss = get_loss(output, y, args, x, stats)
            loss = mse_loss + mass_loss
            running_mass_loss +=mass_loss
        else:
            loss = get_loss(output, y, args, x, stats)
            
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        #tend_out = output[:,:24]-x[:,inds]
        #tend_y = y[:,:24]-x[:,inds]
        #running_tend_loss += torch.mean((tend_out-tend_y)**2)
    loss = running_loss / len(train_data)
    mass_loss = running_mass_loss / len(train_data)
    print('Epoch {}, Train Loss: {:.5f}, mass loss: {:.7f}'.format(epoch+1, loss, mass_loss))

def get_val_loss(model, test_data, epoch, args, stats):
    running_val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_data:
            x = x.to(device)
            output = model(x)
            #print('neg', neg_fraction(x,output))
            #output = torch.sigmoid(output).clone()
            y = y.to(device)
            if args.loss== 'mse_mass':
                loss,mass = get_loss(output, y, args,x, stats)
            else:
                loss= get_loss(output, y, args,x, stats)
            running_val_loss += loss.item()
    loss = running_val_loss/len(test_data)
    print('Epoch {}, Val Loss: {:.5f}'.format(epoch+1, loss))
    model.train()
    return loss
    
def checkpoint(model, val_loss, best, input_dim, output_dim, args):
    if val_loss < best:
        checkpoint = {'model': model,'state_dict': model.state_dict()}
        if not os.path.exists('./models'):
            os.makedirs('./models')
        torch.save(checkpoint, './models/'+args.model_id+'.pth')
        
def check_for_early_stopping(val_loss, best, patience):
    is_stop = False
    if val_loss < best:
        patience = 0
    else:
        patience+=1
    if patience ==5:
        is_stop = True  
    return is_stop, patience

def get_val_scores(model, X_test, y_test, epoch, stats, args):
    mse = 0
    r2 = 0
    mass = 0
    neg = 0
    model.eval()
    pred = model(torch.Tensor(X_test).to(device))
    #pred = torch.sigmoid(pred)
    pred = pred.cpu().detach().numpy()
    inds = [10]+[i for i in range(12,35)]
    #scale back
    if args.scale == 'z':
        pred = standard_transform_y_inv(stats, pred)
        y_test = standard_transform_y_inv(stats, y_test)
        X_test = standard_transform_x_inv(stats, X_test)
    elif args.scale == 'minmax':
        pred = minmax_transform_y_inv(stats, pred)
        y_test = minmax_transform_y_inv(stats, y_test)
        X_test = minmax_transform_x_inv(stats, X_test)
    elif args.scale == 'species_z':
        pred = species_z_transform_y_inv(stats, pred)
        y_test = species_z_transform_y_inv(stats, y_test)
        X_test = standard_transform_x_inv(stats, X_test)
    elif args.scale == 'species_n':
        pred = species_n_transform_y_inv(stats, pred)
        y_test = species_n_transform_y_inv(stats, y_test)
        X_test = standard_transform_x_inv(stats, X_test)
    if args.scale == 'pre_log':
        pred = standard_transform_y_inv(stats, pred)
        y_test = standard_transform_y_inv(stats, y_test)
        X_test = standard_transform_x_inv(stats, X_test)
            
    #tend_z = z(log(x1)-log(x0))
    #exp(z-1(tend_z)+log(x0))= x1
    #tend = x0 - exp(z-1(tend_z)+log(x0)) = 
    if args.scale == 'pre_log':
        X_log = X_test.copy()
        X_test[:,inds] = np.exp(X_test[:,inds])-EPS
        pred[:,:24] = X_unlog - (np.exp(pred[:,:24]+X_log[:,inds])-EPS)
        y_test[:,:24] = X_unlog - (np.exp(y_test[:,:24]+X_log[:,inds])-EPS)
        
        
    scores = get_scores(y_test, pred, X_test, stats)
    

    return scores["R2"], scores["RMSE"], scores["Mass RMSE"], scores["Negative fraction"]

def save_validation_scores(r2, mse, mass, neg, args):
    if not os.path.exists('./data/epoch_scores'):
        os.makedirs('./data/epoch_scores')
    np.save('./data/epoch_scores/'+args.model_id+'_epoch_r2_scores.npy', np.array(r2))
    np.save('./data/epoch_scores/'+args.model_id+'_epoch_rmse_scores.npy', np.array(mse))
    np.save('./data/epoch_scores/'+args.model_id+'_epoch_mass_scores.npy', np.array(mass))
    np.save('./data/epoch_scores/'+args.model_id+'_epoch_neg_scores.npy', np.array(neg))
    
#####
### create evaluation

def create_report(model, X_test, y_test, stats, args):
    #predict
    pred = model(torch.Tensor(X_test).to(device))
    #pred = torch.sigmoid(pred)
    pred = pred.cpu().detach().numpy()
    #scale back
    inds = [10]+[i for i in range(12,35)]
    if args.model == 'classification':
        classes = get_classes(pred)
        np.save('./data/classes.npy',classes)
        scores = get_class_scores(y_test, pred)
        
    else:
        
        if args.scale == 'z':
            pred = standard_transform_y_inv(stats, pred)
        elif args.scale == 'pre_z':
            print('neg', neg_fraction_numpy(X_test,pred,stats))
            X_scale = X_test.copy()
            X_test = standard_transform_x_inv(stats, X_test)
            pred[:,:24] =  pred[:,:24]*stats['xtrain_std'][inds]+stats['xtrain_mean'][inds]
            y_test[:,24:] = standard_transform_y_inv(stats, y_test)[:,24:]
            y_test[:,:24] =  y_test[:,:24]*stats['xtrain_std'][inds]+stats['xtrain_mean'][inds]
            pred[:,24:] = standard_transform_y_inv(stats, pred)[:,24:]
            print('neg after scale',np.mean((pred[:,:24]+X_test[:,inds])<0,axis=0))
            #tend = y-x
            #tend_z = z(y)-z(x)
            #z-1(tendz+z(x)) = y
        elif args.scale == 'pre_log':
            pred = standard_transform_y_inv(stats, pred)
            y_test = standard_transform_y_inv(stats, y_test)
            X_test = standard_transform_x_inv(stats, X_test)
            X_log = X_test.copy()
            X_test[:,inds] = np.exp(X_test[:,inds])-EPS
            pred[:,24:] = np.exp(pred[:,24:])-EPS
            y_test[:,24:] = np.exp(y_test[:,24:])-EPS
            print(pred[:,:24].max(),pred[:,:24].min(),y_test[:,:24].max(),y_test[:,:24].min())
            y_test[:,:24] = (np.exp(y_test[:,:24]+X_log[:,inds])-EPS)-X_test[:,inds] 
            pred[:,:24] = (np.exp(pred[:,:24]+X_log[:,inds])-EPS)- X_test[:,inds]
            #tend_z = z(log(y)-log(x))
            #exp(z-1(tend_z)+log(x))= y
            #tend = exp(z-1(tend_z)+log(x0)) - x
            #tend = y - x 
            
        elif args.scale == 'minmax':
            pred = minmax_transform_y_inv(stats, pred)
        elif args.scale == 'bc':
            pred = bc_transform_x_inv(stats, pred)
        elif args.scale == 'log':
            pred = log_tend_norm_transform_x_inv(stats, pred)
        if args.scale == 'z':
            y_test = standard_transform_y_inv(stats, y_test)
            X_test = standard_transform_x_inv(stats, X_test)
        elif args.scale == 'minmax':
            y_test = minmax_transform_y_inv(stats, y_test)
            X_test = minmax_transform_x_inv(stats, X_test)
        elif args.scale == 'bc':
            y_test = bc_transform_x_inv(stats, y_test)
            X_test[:,11:] = bc_transform_x_inv(stats, X_test[:,11:])
        elif args.scale == 'log':
            y_test = log_tend_norm_transform_y_inv(stats, y_test)
            X_test = log_full_norm_transform_x_inv(stats, X_test)
        if args.log:
            if not os.path.isfile('./data/classes.npy'):
                print('Warning: Class prediction need to be done before running log case, score will be wrong')
            else:
                signs = np.load('./data/classes.npy')
                pred *= signs
                
        if args.model == 'correction':
            pred = correction(pred, X_test[:,8:])
        np.save('./data/prediction.npy',pred)
        scores = get_scores(y_test, pred, X_test, stats, args)
        single_scores = get_single_scores(y_test, pred, X_test, stats, args)

    #print(scores)

    args_dict = vars(args)
    #combine scorees and args dict
    args_scores_dict = {**args_dict , **scores}
    args_scores_dict = {**args_scores_dict , **single_scores}
    args_scores_dict = {**args_scores_dict , **stats}
    save_dict(args_scores_dict, args)
 
                                    
def save_dict(dictionary, args):
    w = csv.writer(open('./scores/'+str(args.mode)+'_'+args.model_id+'.csv', 'w'))
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])
        
#metrics


def norm(x, stats):
    return (x-stats['ytrain_mean'])/stats['ytrain_std']

def masses(y):
    so4_mass = y[:,0]+y[:,1]+y[:,2]+y[:,3]+y[:,4]
    bc_mass = y[:,5]+y[:,6]+y[:,7]+y[:,8]
    oc_mass = y[:,9]+y[:,10]+y[:,11]+y[:,12]
    du_mass = y[:,13]+y[:,14]+y[:,15]+y[:,16]
    return np.array([so4_mass, bc_mass, oc_mass, du_mass])

def norm_rmse(true, pred, stats):
    true_norm = norm(true, stats)
    pred_norm = norm(pred, stats)
    rmse=np.sqrt(mean_squared_error(true_norm, pred_norm))
    return rmse

def mass_middle(pred, stats):
    mass_mean = np.array([stats['so4_mean'],stats['bc_mean'],stats['oc_mean'],stats['du_mean']])
    mass = masses(pred[:,:17])
    mass_means = np.mean(mass, axis=1)/mass_mean
    return mass_means

def mass_rmse(true,pred, stats):
    mass_mean = np.array([stats['so4_mean'],stats['bc_mean'],stats['oc_mean'],stats['du_mean']])
    N = true.shape[0]
    mass_means = np.zeros((4,))
    mass = masses(pred[:,:17])
    for i in range(4):
        mass_means[i] = np.sqrt(mean_squared_error(np.zeros((N,)),mass[i])) 
    return mass_means/mass_mean

def neg_mean(full_pred, stats):
    N = full_pred.shape[0]
    neg_means = np.zeros((24,))
    for i in range(24):
        neg_means[i] = np.sum(full_pred[full_pred[:,i]<0,i], axis=0)/N
    return np.mean(neg_means/stats["means"]) #stats

def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def get_class_scores(y_val_cl, pred_cl):
    TP = np.sum(np.logical_and(y_val_cl==1, pred_cl>0.5), axis=0)
    TN = np.sum(np.logical_and(y_val_cl==0, pred_cl<0.5), axis=0)
    FN = np.sum(np.logical_and(y_val_cl==1, pred_cl<0.5), axis=0)
    FP = np.sum(np.logical_and(y_val_cl==0, pred_cl>0.5), axis=0)
    N = np.sum(y_val_cl==0, axis=0)
    P = np.sum(y_val_cl==1, axis=0)
    acc = (TP+TN)/(N+P)
    prec = TP/(FP+TP)
    recall = TP/(TP+FN)
    return {'Accuracy': acc, 'Precision': prec, 'Recall': recall}

def get_classes(signs):
    y_signs = np.ones((signs.shape[0],28))
    y_signs[:,0] = -1
    y_signs[signs[:,0]<0.5,1] = -1
    y_signs[signs[:,1]<0.5,2] = -1
    y_signs[signs[:,2]<0.5,3] = -1
    y_signs[:,4] = 1
    y_signs[signs[:,3]<0.5,5] = -1
    y_signs[signs[:,4]<0.5,6] = -1
    y_signs[:,7] = 1
    y_signs[:,8] = -1
    y_signs[signs[:,5]<0.5,9] = -1
    y_signs[signs[:,6]<0.5,10] = -1
    y_signs[:,11] = 1
    y_signs[:,12] = -1
    y_signs[signs[:,7]<0.5,13] = -1
    y_signs[:,14] = 1
    y_signs[:,15] = -1
    y_signs[:,16] = -1
    y_signs[signs[:,8]<0.5,17] = -1
    y_signs[signs[:,9]<0.5,18] = -1
    y_signs[signs[:,10]<0.5,19] = -1
    y_signs[:,20] = 1
    y_signs[:,21] = -1
    y_signs[:,22] = -1
    y_signs[:,23] = -1
    return y_signs

def correction(y, x):
    pos = numpy_relu(y[:,:24]+x)
    y[:,24:] = numpy_relu(y[:,24:])
    y[:,:24] = pos - x
    return y


def numpy_relu(x):
    return x * (x > 0)
    
    
