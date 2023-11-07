23.11.05 Bug Fixed with making testset in processing_data.py

23.11.07 Update README about validation. Plan to upload some codes for training with valid/test.

23.11.08 Adding processing_data_valid.py and update training_validation.py for training with validation.

# Resources:

+ README.md: this file.
+ data/davis/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/davis/Y,ligands_can.txt,proteins.txt
  data/kiba/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/kiba/Y,ligands_can.txt,proteins.txt
  These file were downloaded from https://github.com/hkmztrk/DeepDTA/tree/master/data

We don't use this files. Just simply make csv file from YOUR DATASET.


---------


###  Source codes:
+ create_data.py: create data in pytorch format (old version, you don't need to run this code)
+ processing_data.py : create data in pytorch fromat with csv (new version, this will give you input files for training model)
+ utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.
+ training.py: train a GraphDTA model.
+ models/ginconv.py, gat.py, gat_gcn.py, and gcn.py: proposed models GINConvNet, GATNet, GAT_GCN, and GCNNet receiving graphs as input for drugs.

## Step-by-step running


### 0. Install Python libraries needed

```sh
1. conda create -n graphDTA python==3.9.12
2. conda activate graphDTA
3. pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
4. pip install rdkit-pypi==2021.3.5.1
5. pip install torch_geometric
6. pip install pandas==1.5.3
7. pip install networkx==2.8.8
```
You should check your CUDA driver version.

Mine is CUDA==11.4 so this worked for me, but if your version is different, fix the line 3.


### 1. Create csv data & input dataset for model.
The old version of graphDTA used create_data.py for making input csv file along with Dataset of DeepDTA. (https://github.com/thinng/GraphDTA/issues/8#issuecomment-759070065)
But it's too hassle to use this code for now. We want to use graphDTA with our own datasets not the DeepDTA datasets.
So, skip this create_data.py.

Rather, we use processing_data.py which I made new. This code is without making triple formated csv file with DeepDTA datasets.
For running this code, you have to make the CSV file with this format.


| smiles  | sequence | pka | label |
| ------------- | ------------- |------------- |------------- |
| COc1cc(CCCOC(=O)  | MDVLSPGQGNNTTS  |10.34969248 | 1 |
| OC(=O)C=C | MSWATRPPF  |5.568636236 | 0

Where smiles is compound sequences, sequence is protein AA sequences, pka is affinity values, and label is binary column for binary classification.
It's trivial to add label column with pka values. You should check your dataset and set proper threshold value for making label column.
In my case, I used BindingDB dataset from DeepAffinity and the pka was given with Ki value. I set the threshold value as 6.

Make these two csv files '{bindingdb}_train.csv' and '{bindingdb}_test.csv'.

Note that, {bindingdb} can be changed along your own dataset, but you should fix the 61 (for dt_name in ['davis', 'kiba', 'bindingdb']:)
and 72 (datasets = ['davis','kiba','bindingdb']) lines into your own dataset.
Move those two csv files under GraphDTA/data/

If you want to add Valid dataset, then read 3. Train a prediction model with validation dataset section below.
And I highly recommend it.

Then simply run 
```
cd YOUR/PATH/TO/graphDTA
python processing_data.py
```

This will give you tarin/test.pt files for running/training your model.

### 2. Train a prediction model
To train a model using training data. The model is chosen if it gains the best MSE/AUC for testing data.

Running 

```sh
conda activate graphDTA
python training.py 0 0 0 0
```


where the first argument is for the index of the datasets, 0/1/2 for 'davis', 'kiba' and 'bindingdb' respectively;
 the second argument is for the index of the models, 0/1/2/3 for GINConvNet, GATNet, GAT_GCN, or GCNNet, respectively;
 the third argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1', respectively.;
 Note that your actual CUDA name may vary from these, so please change the following code accordingly:
```sh
cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
```
 The last argument is for the model mode 0 / 1 for 'classification' and 'regression', respectively;

This returns the model and result files for the modelling achieving the best MSE/AUC for testing data throughout the training.
For example, it returns two files regression_model_GATNet_davis.model and regression_result_GATNet_davis.csv when running regression model with GATNet on Davis data.
And if you do it with classification model, it returns two files classification_model_GATNet_davis.model and classification_result_GATNet_davis.csv.

### 3. Train a prediction model with validation dataset
! My method is slightly different from original GraphDTA.

! They split train/test data into train/valid/test during running training, but our method explicitly needs train/valid/test.pt for running.

In "3. Train a prediction model", a model is trained on training data and chosen when it gains the best MSE for valid data.

As I mentioned above, simply make three csv files('{bindingdb}_train.csv', '{bindingdb}_valid.csv and '{bindingdb}_test.csv'.) with same format 
and move those three csv files under GraphDTA/data/

Then simply run 
```
cd YOUR/PATH/TO/graphDTA
python processing_data_valid.py
```
This will retrun train/valid/test.pt as input dataset for training.

### 4. Train a prediction model with validation
Using same arguments. The arguments are explained above.

! updated the first argument 0 / 1 for 'toy_bindingdb' and 'bindingdb', respectively. others are same.

```sh
python training_validation.py 0 0 0 0
```

This returns the model achieving the best MSE for validation data throughout the training and performance results of the model on testing data.
Note that our method is different from original graphDTA, you can't use davis/kiba dataset if you want to do train with validation since there are no davis/kiba_valid.pt.
I attached toy_bindingdb train/valid/test dataset.
