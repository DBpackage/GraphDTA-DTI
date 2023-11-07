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
from models.gcn_test import GCNNet_TEST
from utils import *
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import warnings

# 경고메세지 끄기
warnings.filterwarnings(action='ignore')


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, mode):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    if mode == 'regression':
        #loss_fn = nn.MSELoss()
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            #print("regression output", output[:10], output.size()); input()
            #print("classification data.y.view(-1, 1)",data.y.view(-1, 1)[:10]); input()
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
        #loss_fn = nn.BCELoss()
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            #print("classification output", output[:10], output.size()); input()
            #print("classification data.y_c.view(-1, 1)",data.y_c.view(-1, 1)[:10]); input()
            loss = loss_fn(output, data.y_c.view(-1, 1).float().to(device)) # classification : nn.BCEWithLogitsLoss()
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
        return total_labels.numpy().flatten(),total_preds.numpy().flatten()
    elif mode == 'classification':
        with torch.no_grad():
            for data in tqdm(loader):
                data = data.to(device)
                output = model(data)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, data.y_c.view(-1, 1).cpu()), 0)
                
        return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = [['toy_bindingdb', 'bindingdb'][int(sys.argv[1])]] #
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet, GCNNet_TEST][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
print('cuda_name:', cuda_name)

mode = ['classification', 'regression'][int(sys.argv[4])] #

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_valid = 'data/processed/' + dataset + '_valid.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')
        valid_data = TestbedDataset(root='data', dataset=dataset+'_valid')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)

        if mode == 'regression':
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_valid_mse = 1000
            best_test_mse  = 1000
            best_test_ci   = 0
            best_test_rmse = 0
            best_test_pearson = 0
            best_test_spearman = 0
            best_epoch = -1
            model_file_name = 'regression_model_' + model_st + '_' + dataset +'.model' 
            result_file_name = 'regression_result_' + model_st + '_' + dataset +  '.csv'
            for epoch in range(NUM_EPOCHS):
                train(model, device, train_loader, optimizer, epoch+1, mode=mode)
                G,P = predicting(model, device, valid_loader, mode=mode) # Ground Truth, Predicted Value
                # validation first
                print("predicting for valid data")
                val = mse(G,P)
                if val < best_valid_mse:
                    best_valid_auc = val
                    torch.save(model.state_dict(), model_file_name)

                print('predicting for test data')
                G,P = predicting(model, device, test_loader, mode=mode)
                ret = [rmse(G,P), mse(G,P), pearson(G,P), spearman(G,P), ci_new(G,P)];
                if ret[1] < best_test_mse:
                    with open(result_file_name,'w') as f:
                        f.write(','.join(map(str,ret)))
                    best_epoch = epoch+1
                    best_test_mse  = ret[1]
                    best_test_ci   = ret[-1]
                    best_test_rmse = ret[0]
                    best_test_pearson  = ret[2]
                    best_test_spearman = ret[3]
                    print('mse(', best_test_mse,')improved at epoch :', best_epoch, '; best_rmse, best_ci, best_pearson, best_spearman :', \
                            round(best_test_rmse, 6), round(best_test_ci,6), round(best_test_pearson,6), round(best_test_spearman,6), \
                            model_st, dataset)
                    # print("Best RMSE : {0}, MSE : {1}, CI : {2}, pearson : {3}, spearman : {4}".format(ret[0], ret[1], ret[-1], ret[2], ret[3]))
                else:
                    print('Cur MSE :', ret[1],'No improvement since epoch :', best_epoch )
                    # print("Best RMSE : {0}, MSE : {1}, CI : {2}, pearson : {3}, spearman : {4}".format(ret[0], ret[1], ret[-1], ret[2], ret[3]))

        elif mode == 'classification':
            loss_fn = nn.BCEWithLogitsLoss()
            #loss_fn = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_valid_auc = 0
            best_test_auc  = 0
            best_test_prc = 0
            best_test_acc = 0
            # best_test_R = 0
            # best_test_P = 0
            best_epoch = -1
            model_file_name = 'classification_model_' + model_st + '_' + dataset +  '.model'
            result_file_name = 'classification_result_' + model_st + '_' + dataset +  '.csv'
            for epoch in range(NUM_EPOCHS):
                train(model, device, train_loader, optimizer, epoch+1, mode=mode)
                print("predicting for valid data")
                G,P = predicting(model, device, valid_loader, mode=mode)
                val = roc_auc_score(G,P)
                if val > best_valid_auc:
                    best_valid_auc = val
                    torch.save(model.state_dict(), model_file_name)
                
                print('predicting for test data')
                G,P = predicting(model, device, test_loader, mode=mode) # Ground Truth, Probabilities
                ret = [roc_auc_score(G, P), prc_auc(G,P), accuracy(G,P)]

                if ret[0] > best_test_auc:
                    best_test_auc = ret[0]
                    with open(result_file_name,'w') as f:
                        f.write(','.join(map(str,ret)))
                    best_epoch = epoch + 1
                    # best_auc = ret[0]
                    best_test_prc = ret[1]
                    best_test_acc = ret[2]
                    print('AUC(',best_test_auc,')improved at epoch :', best_epoch, '; best_PRC, best_ACC:', best_test_prc, best_test_acc, model_st, dataset)
                else:
                    print("Cur AUC :", ret[0],' No improvement since epoch :', best_epoch)

            print("BEST AUC, PRC, ACC, EPOCH : ",best_test_auc, best_test_prc, best_test_acc, best_epoch)
