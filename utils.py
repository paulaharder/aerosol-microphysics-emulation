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
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import csv
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#######arguments

def add_nn_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='train', help="train or eval with test")
    parser.add_argument("--signs", default=False, help="needed for log with mass reg.")
    parser.add_argument("--scale", default='z', help="z or log")
    parser.add_argument("--model", default="standard", help="standard, completion, correction, positivity, standard_log, log_mass, classification")
    parser.add_argument("--model_id", default="standard_test")
    parser.add_argument("--log", default=False)
    parser.add_argument("--lr", default=0.001, help="learning rate")
    parser.add_argument("--width", default=128, help="width of hidden layers")
    parser.add_argument("--loss", default='mse')
    parser.add_argument("--optimizer", default='adam')
    parser.add_argument("--weight_decay", default=1e-9)
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--epochs", default=1)
    parser.add_argument("--early_stop", default=False)
    parser.add_argument("--save_val_scores", default=False)
    return parser.parse_args()

########
###stats
def calculate_stats(X_train, y_train, X_test, y_test, args):
    means = np.mean(y_test[:,:24]+X_test[:,8:], axis=0)
    X_log_eps = log_transform(X_train)
    y_log_eps = log_transform(y_train)
    
    so4 = np.concatenate((X_train[:,8],X_train[:,9],X_train[:,10],X_train[:,11],X_train[:,12]), axis=0)
    bc = np.concatenate((X_train[:,13],X_train[:,14],X_train[:,15],X_train[:,16]),axis=0)
    oc = np.concatenate((X_train[:,17],X_train[:,18],X_train[:,19],X_train[:,20]),axis=0)
    du = np.concatenate((X_train[:,21],X_train[:,24],X_train[:,22],X_train[:,23]),axis=0)

    so4_mean = np.mean(so4)
    bc_mean = np.mean(bc)
    oc_mean = np.mean(oc)  
    du_mean = np.mean(du) 
    
    stats = {'xtrain_mean': np.mean(X_train, axis=0),
            'xtrain_std': np.std(X_train, axis=0),
            'ytrain_mean': np.mean(y_train, axis=0),
            'ytrain_std': np.std(y_train, axis=0),
            'so4_mean':so4_mean,
            'bc_mean':bc_mean,
            'oc_mean':oc_mean,
            'du_mean':du_mean,
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
        model = Base(in_features, out_features, args.width)
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

def get_loss(output, y, args):
    criterion = nn.MSELoss()
    class_criterion = nn.BCELoss()
    if args.loss == 'mse':
        return criterion(output, y)
    elif args.loss == 'mse_mass':
        return criterion(output[:,:28], y[:,:28])+overall_z_mass(output)
    elif args.loss == 'mse_positivity':
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

##############
########losses

def overall_z_mass(y):  
    so4 = torch.mean(mass_so4(y[:,:5]))
    bc = torch.mean(mass_bc(y[:,5:9]))
    oc = torch.mean(mass_oc(y[:,9:13]))
    du = torch.mean(mass_du(y[:,13:17]))
    mass = so4+bc+oc+ du    
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
        model_step(model, train_data, optimizer, i, args)
        val_loss = get_val_loss(model, test_data, i, args)
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
                           
def model_step(model, train_data, optimizer, epoch, args):
    running_loss = 0
    for x, y in train_data:
        x = x.to(device)
        optimizer.zero_grad()
        model.to(device)
        output = model(x)
        y = y.to(device)
        loss = get_loss(output, y, args)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss = running_loss / len(train_data)
    print('Epoch {}, Train Loss: {:.5f}'.format(epoch+1, loss))

def get_val_loss(model, test_data, epoch, args):
    running_val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_data:
            x = x.to(device)
            output = model(x)
            y = y.to(device)
            loss = get_loss(output, y, args)
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
    pred = pred.cpu().detach().numpy()
    #scale back
    
    pred = standard_transform_y_inv(stats, pred)
    y_test = standard_transform_y_inv(stats, y_test)
    X_test = standard_transform_x_inv(stats, X_test)
    
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
    pred = pred.cpu().detach().numpy()
    #scale back
    if args.model == 'classification':
        classes = get_classes(pred)
        np.save('./data/classes.npy',classes)
        scores = get_class_scores(y_test, pred)
        
    else:
        if not args.model == 'correction':
            if args.scale == 'z':
                pred = standard_transform_y_inv(stats, pred)
            elif args.scale == 'log':
                pred = log_tend_norm_transform_y_inv(stats, pred)
            
        np.save('./data/prediction.npy',pred)
        if args.scale == 'z':
            y_test = standard_transform_y_inv(stats, y_test)
            X_test = standard_transform_x_inv(stats, X_test)
        elif args.scale == 'log':
            y_test = log_tend_norm_transform_y_inv(stats, y_test)
            X_test = log_full_norm_transform_x_inv(stats, X_test)
        if args.log:
            if not os.path.isfile('./data/classes.npy'):
                print('Warning: Class prediction need to be done before running log case, score will be wrong')
            else:
                signs = np.load('./data/classes.npy')
                pred *= signs

        scores = get_scores(y_test, pred, X_test, stats)

    print(scores)

    args_dict = vars(args)
    #combine scorees and args dict
    args_scores_dict = args_dict | scores
    #save dict
    save_dict(args_scores_dict, args)
 
                                    
def save_dict(dictionary, args):
    w = csv.writer(open('./data/'+str(args.mode)+'_'+args.model_id+'.csv', 'w'))
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
    return np.mean(mass_means/mass_mean)

def neg_mean(full_pred, stats):
    N = full_pred.shape[0]
    neg_means = np.zeros((24,))
    for i in range(24):
        neg_means[i] = np.sum(full_pred[full_pred[:,i]<0,i], axis=0)/N
    return np.mean(neg_means/stats["means"]) #stats

def get_scores(true, pred, X_test, stats):
    r2=  r2_score(true, pred)
    r2_log =  r2_score(np.log(np.abs(true)+1e-8), np.log(np.abs(pred)+1e-8))
    rmse =  norm_rmse(true, pred, stats) #take from standardized values
    mass_biases = mass_middle(pred, stats) #individual masses divided by the respective abs mean
    masses_rmse = mass_rmse(true, pred, stats)#individual masses divided by the respective abs mean
    full_pred = pred[:,:24]+X_test[:,8:]
    #print(full_pred[:10,0],pred[:10,0],X_test[:10,8])
    #print(full_pred.min())
    #full_pred = np.concatenate((full_pred, pred[:,24:]), axis=1)
    neg_frac = np.zeros((24,1))
    for i in range(24):
        neg_frac[i] = np.mean(full_pred[:,i]<0)
    neg_frac2 = np.mean(pred[:,24:]<0)
    negative_mean = neg_mean(full_pred, stats)
    return {'R2': r2, 'R2 log': r2_log, 'RMSE': rmse, 'Mass Bias':mass_biases, 'Mass RMSE':masses_rmse,'Negative fraction': neg_frac,'Negative fraction2': neg_frac2,'Negative mean' : negative_mean}


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
    
    
