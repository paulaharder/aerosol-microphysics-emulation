"""
Description : Train neural network model to predict one time step of M7
Options:
  --dataset=<data_set_id>
  --tendency=<predict_tendency_or_not>
  --scale_in=<input_scaler>
  --scale_out=<output_scaler>
"""
import numpy as np
from utils import get_scaler_x, get_scaler_y, get_model, train_model, create_report
import torch.nn as nn 
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    #loading data
    print('loading data...')    
    X_train = np.load('./data/'+args.dataset+'/X_train.npy')
    y_train = np.load('./data/'+args.dataset+'/y_train.npy')
    if args.train:
        X_test = np.load('./data/'+args.dataset+'/X_val.npy')
        y_test = np.load('./data/'+args.dataset+'/y_val.npy')
    else:
        X_test = np.load('./data/'+args.dataset+'/X_test.npy')
        y_test = np.load('./data/'+args.dataset+'/y_test.npy')
    
    y_train[:,:26] -= X_train[:,8:]
    y_test[:,:26] -= X_test[:,8:]
    y_train_sign = np.sign(y_train)
    y_test_sign = np.sign(y_test)
    
    #transform
    scaler_in = get_scaler_x(X_train, args.scale_in)
    scaler_out = get_scaler_y(y_train_t, args.scale_out)
    X_train = scaler_in.transform(X_train)
    y_train = scaler_out.transform(y_train)
    X_test = scaler_in.transform(X_test)
    y_test= scaler_out.transform(y_test)
    
    if args.signs:
        X_train = np.concatenate((X_train, y_train_signs), axis=1)
        X_test = np.concatenate((X_test, y_test_signs), axis=1)
    
    train_data = create_dataloader(X_train, y_train, args)
    val_data = create_test_dataloader(X_test, y_test, args)
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = get_model(in_features=input_dim, out_features=output_dim, args)
    
    if train:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.MSELoss()
        train_model(model=model, train_data=train_data, test_data=val_data, criterion=criterion, optimizer=optimizer, input_dim=input_dim, output_dim=output_dim, args=args)
    
    model = get_model(in_features=input_dim, out_features=output_dim, args)
    name = args.model_id
    model.load_state_dict(torch.load('./models/'+name+'.pth') ['state_dict'])
    model.to(device)
    model.eval()
    create_report(model, X_test, y_test, args)
    
if __name__ == "__main__":
    args = add_nn_arguments()
    main(args)