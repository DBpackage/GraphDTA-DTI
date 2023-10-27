import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, mode):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    if mode == 'regression':
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print("regression output", output[:5], output.size()); input()
            loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) # regression : nn.MSELoss()
            loss.backward()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    elif mode == 'classification':
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print("classification output", output[:5], output.size()); input()
            #print("classification data.y_c", data.y_c.view(-1, 1), data.y_c.view(-1,1).size()); input()
            #print("classification data.y_c.view(-1, 1)",data.y_c.view(-1, 1)); input()
            loss = loss_fn(output, data.y_c.view(-1, 1).float().to(device)) # classification : nn.BCEWithLogitsLoss()
            #print("loss :",loss)
            #print("loss:",loss.item(), loss.size()); input()
            loss.backward()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    else:
        pass


def predicting(model, device, loader, mode):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    if mode == 'regression':
        with torch.no_grad():
            for data in tqdm(loader):
                data = data.to(device)
                output = model(data)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
        return total_labels.numpy().flatten(), total_preds.numpy().flatten()
    elif mode == 'classification':
        with torch.no_grad():
            for data in tqdm(loader):
                data = data.to(device)
                output = model(data)
                # output = nn.Sigmoid()(output)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, data.y_c.view(-1, 1).cpu()), 0)

            total_preds = nn.Sigmoid()(total_preds)
                
        return total_labels.numpy().flatten(), total_preds.numpy().flatten()

datasets = [['davis','kiba','bindingdb'][int(sys.argv[1])]] #
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

mode = ['classification', 'regression'][int(sys.argv[4])] #

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 40
NUM_EPOCHS = 500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')
        #print("train_data", train_data); input()
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        if mode == 'regression':
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_mse = 1000
            best_ci = 0
            best_epoch = -1
            model_file_name = 'regression_model_' + model_st + '_' + dataset +  '.model'
            result_file_name = 'regression_result_' + model_st + '_' + dataset +  '.csv'
            for epoch in range(NUM_EPOCHS):
                train(model, device, train_loader, optimizer, epoch+1, mode=mode)
                
                G,P = predicting(model, device, test_loader, mode=mode) # Ground Truth, Predicted Float Value
                ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]; 
                
                if ret[1]<best_mse:
                    torch.save(model.state_dict(), model_file_name); 
                    with open(result_file_name,'w') as f:
                        f.write(','.join(map(str,ret)))
                    best_epoch = epoch+1
                    best_mse = ret[1]
                    best_ci = ret[-1]
                    print('rmse improved at epoch ', best_epoch, '; best_mse, best_ci,', best_mse,best_ci,model_st,dataset)
                    print("Best RMSE : {0}, MSE : {1}, CI : {2}, pearson : {3}, spearman : {4}".format(ret[0], ret[1], ret[-1], ret[2], ret[3]))
                else:
                    print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
                    print("Best RMSE : {0}, MSE : {1}, CI : {2}, pearson : {3}, spearman : {4}".format(ret[0], ret[1], ret[-1], ret[2], ret[3]))

        elif mode == 'classification':
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_auc = 0
            best_prc = 0
            best_acc = 0
            best_epoch = -1
            model_file_name = 'classification_model_' + model_st + '_' + dataset +  '.model'
            result_file_name = 'classification_result_' + model_st + '_' + dataset +  '.csv'
            for epoch in range(NUM_EPOCHS):
                train(model, device, train_loader, optimizer, epoch+1, mode=mode)
                G, P = predicting(model, device, test_loader, mode=mode) # Ground Truth, Predicted Probabilities
                # print("Ground Truth :", G[22132:22137]); 
                # print("Predicted    :", P[22132:22137]);
                
                ret = [roc_auc_score(G,P), prc_auc(G,P), accuracy(G,P)]
                if ret[0]>best_auc:
                    torch.save(model.state_dict(), model_file_name)
                    with open(result_file_name,'w') as f:
                        f.write(','.join(map(str,ret)))
                    best_epoch = epoch+1
                    best_auc = ret[0]
                    best_prc = ret[1]
                    best_acc = ret[2]
                    print('AUC(',best_auc, ')improved at epoch ', best_epoch, '; best_PRC, best_ACC:', best_prc, best_acc, model_st, dataset)
                else:
                    print(ret[0],'No improvement since epoch ', best_epoch, '; best_PRC, best_ACC:', best_prc, best_acc, model_st, dataset)
