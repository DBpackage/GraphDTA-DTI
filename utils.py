import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss, roc_curve, precision_score, recall_score

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, y_c=None, transform=None,
                 pre_transform=None, smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, y_c, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # y_c: list of labels for DTI classification
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y, y_c, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            class_labels = y_c[i]
            #print("process: class_labels", class_labels)
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]),
                                y_c=torch.LongTensor([class_labels]))
            GCNData.target = torch.LongTensor([target])
            #GCNData.y_c    = torch.LongTensor([class_labels])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        #print("process:data_list:",data_list);input()
        #print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        #print("process:data", data);input()
        #print("process:slices", slices);input()
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f): # I optimzed the CI calculation function. It reduced almost 5 times with same result.
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    n = len(y)
    c, d = 0, 0
    z = 0.0
    S = 0.0

    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[i] != y[j]:
                z += 1
                if f[i] < f[j]:
                    S += 1
                elif f[i] == f[j]:
                    S += 0.5
    if z > 0:
        ci = S / z
    else:
        ci = 0.0

    return ci

def prc_auc(targets, preds):
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    fpr, tpr, thresholds = roc_curve(targets, preds)

    J = tpr - fpr
    ix = np.argmax(J)
    thred_optim = thresholds[ix]

    y_pred_s = [1 if i else 0 for i in (preds >= thred_optim)]

    print('Recall    : ', recall_score(targets, y_pred_s))
    print('Precision : ', precision_score(targets, y_pred_s))

    precision, recall, _ = precision_recall_curve(targets, preds)
    #print("precision", precision)
    #print("recall", recall)
    return auc(recall, precision)

def accuracy(targets, preds) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed accuracy.
    """

    fpr, tpr, thresholds = roc_curve(targets, preds)

    J = tpr - fpr
    ix = np.argmax(J)
    thred_optim = thresholds[ix]

    y_pred_s = [1 if i else 0 for i in (preds >= thred_optim)]

   # if type(preds[0]) == list:  # multiclass
    #    hard_preds = [p.index(max(p)) for p in preds]
    #else:
     #   hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction

    return accuracy_score(targets, y_pred_s)
