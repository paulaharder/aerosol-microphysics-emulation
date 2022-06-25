import numpy as np
import sys
sys.path.insert(1, '../')
from utils import standard_transform_x, standard_transform_y, get_model, train_model, create_report, calculate_stats, add_nn_arguments, log_full_norm_transform_x, log_tend_norm_transform_y, create_dataloader, create_test_dataloader
import torch.nn as nn 
import torch
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--device", default='gpu', help="cpu, cpu_single, gpu, pure_gpu")
parser.add_argument("--scale", default='z', help="z or log")
parser.add_argument("--model", default="standard", help="standard, completion, correction, positivity, standard_log, log_mass, classification")
parser.add_argument("--model_id", default="standard_2depth_128width")
parser.add_argument("--width", default=128, help="width of hidden layers")
parser.add_argument("--old_data",default=False)

def main(args):
    
    #set device
    if args.device == 'cpu':
        device = "cpu"
    elif args.device == 'cpu_single':
        device = "cpu"
        torch.set_num_threads(1)
    else:
        device = 'cuda:0'

    #load data
    X_train = np.load('../data/aerosol_emulation_data/X_train.npy')
    y_train = np.load('../data/aerosol_emulation_data/y_train.npy') 
    X_test = np.load('../data/aerosol_emulation_data/X_test.npy')
    y_test = np.load('../data/aerosol_emulation_data/y_test.npy')
    
    if args.old_data:
        X_train = np.delete(X_train, [21,22],axis=1) 
        X_test = np.delete(X_test, [21,22],axis=1)
        
    stats = calculate_stats(X_train, y_train, X_test, y_test, args)
    
    #load model
    checkpoint = torch.load('../models/'+args.model_id+'.pth')
    input_dim = X_test.shape[1]
    output_dim = y_test.shape[1]
    
    model = get_model(in_features=input_dim, out_features=output_dim, args=args, constraints_active=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print('#params:', sum(p.numel() for p in model.parameters()))
    
    #run inference
    
    X_val_ = torch.Tensor(X_test[:192*31*96,:])
    if args.device == 'gpu':
        t_mean = torch.Tensor(stats['ytrain_mean']).to(device)
        t_std = torch.Tensor(stats['ytrain_std']).to(device)
        x_mean = torch.Tensor(stats['xtrain_mean']).to(device)
        x_std = torch.Tensor(stats['xtrain_std']).to(device)
        times = np.zeros((11,6))
        for i in range(11):
            times[i,0] = time.time()
            X_val = X_val_.to(device)
            times[i,1] = time.time()
            X_val = (X_val- x_mean)/x_std
            times[i,2] = time.time()
            prediction = model(X_val)
            times[i,3] = time.time()
            prediction = prediction*t_std + t_mean

            times[i,4] = time.time()
            prediction.cpu()
            times[i,5] = time.time()
        totals = times[1:,5] - times[1:,0]
        pred_only = times[1:,3] - times[1:,2]
        norm_incl = times[1:,4] - times[1:,1]
        print('total:',np.mean(totals), 'gpu pure:', np.mean(norm_incl))
    else:
        times = np.zeros((11,2))
        for i in range(11):
            times[i,0] = time.time()
            X_val = (X_val- stats['xtrain_mean'])/stats['xtrain_std']
            prediction = model(X_val)
            prediction = prediction*stats['ytrain_std'] + stats['ytrain_mean']
            times[i,1] = time.time() 
        totals = times[1:,1] - times[1:,0]
        print(np.mean(times))
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)