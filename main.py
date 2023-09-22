"""
Description : Train neural network model to predict one time step of M7
Options:

  --signs=<need_extra_signs_for_log_mass>
  --classification=<train_classification_net>
  --scale=<scaler>
  --model=<model_version>
"""
import numpy as np
from utils import standard_transform_x, standard_transform_y, bc_transform_x, minmax_transform_x,minmax_transform_y, get_model, train_model, create_report, calculate_stats, add_nn_arguments, log_full_norm_transform_x, log_tend_norm_transform_y, create_dataloader, create_test_dataloader, species_z_transform_y, species_n_transform_y
import torch.nn as nn 
import torch
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    #loading data
    print('loading data...')    
    
    X_train = np.load('../keras-emulator/data/'+args.dataset+'/X_train.npy')
    y_train = np.load('../keras-emulator/data/'+args.dataset+'/y_train.npy')  
    
    if args.mode == 'train':
        X_test = np.load('../keras-emulator/data/'+args.dataset+'/X_val.npy')
        y_test = np.load('../keras-emulator/data/'+args.dataset+'/y_val.npy')
    else:
        X_test = np.load('../keras-emulator/data/'+args.dataset+'/X_test.npy')
        y_test = np.load('../keras-emulator/data/'+args.dataset+'/y_test.npy')
    
    inds = [10]+[i for i in range(12,35)]
    
    eps = 1e-20
    
    #add dh2s04
    X_train[:,10] += 600*X_train[:,11]
    X_test[:,10] += 600*X_test[:,11]
    X_train[X_train[:,10]<0,10]=0
    X_test[X_test[:,10]<0,10]=0
    
    stats = calculate_stats(X_train, y_train, X_test, y_test, args)#
    print(np.mean(y_train<0,axis=0))
    print(np.mean(y_test<0,axis=0))
    print(np.mean(X_train[:,inds]<0,axis=0))
    print(np.mean(X_test[:,inds]<0,axis=0))
    #pre change transform
    if args.scale == 'pre_log':
        #do not log everything, only masses and numbers
        X_train[:,inds] = np.log(X_train[:,inds]+eps)
        X_test[:,inds] = np.log(X_test[:,inds]+eps)
        y_train = np.log(y_train+eps)
        y_test = np.log(y_test+eps)
    elif args.scale == 'pre_z':
        X_train = standard_transform_x(stats, X_train)
        X_test = standard_transform_x(stats, X_test)
        if not args.model == 'classification':
            y_test= standard_transform_y(stats, y_test)
            y_train = standard_transform_y(stats, y_train)
        
    if args.tend_full == 'tend':
        y_train[:,:24] -= X_train[:,inds]
        y_test[:,:24] -= X_test[:,inds]

        
    if args.single == 'bc':
        y_train = y_train[:,5:9]
        y_test = y_test[:,5:9]
        X_train = np.concatenate((X_train[:,:11],X_train[:,16:20]),axis=1)
        X_test = np.concatenate((X_test[:,:11],X_test[:,16:20]),axis=1)
    
    #y_train = y_train[:,:24]
    #y_test = y_test[:,:24]
    
    
    
    
    if args.signs:
        y_train_sign = np.sign(y_train)
        y_test_sign = np.sign(y_test)
        
    if args.model == 'classification':
        inds = [1,2,3,5,6,9,10,13,17,18,19]
        y_train_signs = np.ones_like(y_train)
        y_test_signs = np.ones_like(y_test)
        y_train_signs[y_train<0] = 0
        y_test_signs[y_test<0] = 0
        y_train = y_train_signs[:,inds]
        y_test = y_test_signs[:,inds]

    #transform
    if args.scale == 'z':
        X_train = standard_transform_x(stats, X_train)
        X_test = standard_transform_x(stats, X_test)
        if not args.model == 'classification':
            y_test= standard_transform_y(stats, y_test)
            y_train = standard_transform_y(stats, y_train)
    elif args.scale == 'minmax':
        X_train = minmax_transform_x(stats, X_train)
        X_test = minmax_transform_x(stats, X_test)
        if not args.model == 'classification':
            y_test= minmax_transform_y(stats, y_test)
            y_train = minmax_transform_y(stats, y_train)
    elif args.scale == 'species_z':
        X_train = standard_transform_x(stats, X_train)
        X_test = standard_transform_x(stats, X_test)
        y_test= species_z_transform_y(stats, y_test)
        y_train = species_z_transform_y(stats, y_train)
    elif args.scale == 'species_n':
        X_train = standard_transform_x(stats, X_train)
        X_test = standard_transform_x(stats, X_test)
        y_test= species_n_transform_y(stats, y_test)
        y_train = species_n_transform_y(stats, y_train)
    elif args.scale == 'bc':
        X_train[:,:11] = standard_transform_x(stats, X_train[:,:11])
        X_test[:,:11] = standard_transform_x(stats, X_test[:,:11])
        X_train[:,11:] = bc_transform_x(stats, X_train[:,11:])
        X_test[:,11:] = bc_transform_x(stats, X_test[:,11:])
        if not args.model == 'classification':
            y_test= bc_transform_x(stats, y_test)
            y_train = bc_transform_x(stats, y_train)
    elif args.scale == 'log':
        X_train = log_full_norm_transform_x(stats, X_train)
        X_test = log_full_norm_transform_x(stats, X_test)    
        if not args.model == 'classification':
            y_train = log_tend_norm_transform_y(stats, y_train)
            y_test= log_tend_norm_transform_y(stats, y_test)
    if args.signs:
        X_train = np.concatenate((X_train, y_train_signs), axis=1)
        X_test = np.concatenate((X_test, y_test_signs), axis=1)
    if args.scale == 'pre_log':
        X_train = standard_transform_x(stats, X_train)
        X_test = standard_transform_x(stats, X_test)
        if not args.model == 'classification':
            y_test= standard_transform_y(stats, y_test)
            y_train = standard_transform_y(stats, y_train)
        
        
    print(y_train.shape, y_test.shape)
    
    train_data = create_dataloader(X_train, y_train, args)
    test_data = create_test_dataloader(X_test, y_test, args)
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = get_model(in_features=input_dim, out_features=output_dim, args=args, constraints_active=True)  
    if args.mode == 'train':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_model(model=model, train_data=train_data, test_data=test_data, optimizer=optimizer, input_dim=input_dim, output_dim=output_dim, stats=stats, X_test=X_test, y_test=y_test, args=args)
    
    model = get_model(in_features=input_dim, out_features=output_dim, args=args, constraints_active=True) 
    model.load_state_dict(torch.load('./models/'+args.model_id+'.pth') ['state_dict'])
    model.to(device)
    model.eval()
    create_report(model, X_test, y_test, stats, args)
    
if __name__ == "__main__":
    args = add_nn_arguments()
    main(args)
    
    
