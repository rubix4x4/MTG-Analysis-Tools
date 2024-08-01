# The Purpose of this is to create and train the NN model
import os
import sys
sys.path.append(os.getcwd() + "\\Lib")
import json

# Torch Imports
import torch                                    # Pytorch Library
import torch.nn as nn                           # Neural Network Modules and Loss Functions
import torch.optim as optim                     # Optimization algorithms
import torch.nn.functional as Functional        # Functions with no parameters
from torch.utils.data import DataLoader         # Easier data set management

# Pre Network Data Manipulation
import pandas as pd
import numpy as np
import ParseFunctions as PF
import sklearn.preprocessing as SKPRE
import matplotlib.pyplot as plt

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,num_classes)
        
    def forward(self,x): # This basically describes how we move from one layer to the next
        x = Functional.relu(self.fc1(x))
        x = Functional.relu(self.fc2(x))
        x = Functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Allows me to construct my own dataset from my own tensors
class CustomDataSet:
    def __init__(self,features,labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self,idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

# Load PandaDataSet
f = open('Parsed Data Sets/FullPostOracleAnalysis.json', encoding='utf8')
data = json.load(f)
Data = pd.DataFrame.from_records(data)
DataNN = Data
DataColumns = Data.columns
ColumnsToKeep = ['cmc','edhrec_rank', 'power',
       'toughness','Creature', 'Enchantment', 'Sorcery',
       'Instant', 'Artifact', 'Battle', 'Planeswalker', '{W}', '{U}', '{B}',
       '{R}', '{G}', '{C}', 'LABEL_0', 'LABEL_1', 'LABEL_2', 'score']

# Remove Columns Unnecessary for Analysis
for Column in DataColumns:
    if Column not in ColumnsToKeep:
        DataNN = DataNN.drop(Column,axis=1)

# Fix Power and Toughness Values
DataNN['toughness'] = DataNN['toughness'].apply(PF.PowToughFix)
DataNN['power'] = DataNN['power'].apply(PF.PowToughFix)
DataNNCol = DataNN.columns
# Plot histograms

# Scale all values in DataNN using minmax scaler
scaler = SKPRE.MinMaxScaler()
DataNN[DataNNCol] = scaler.fit_transform(DataNN[DataNNCol])
labels = torch.tensor(DataNN['edhrec_rank']).type(torch.float)

FeatureDF = DataNN.drop('edhrec_rank',axis=1)
FeatureNP = FeatureDF.values
FeatureNP = FeatureNP.astype(float)
features = torch.tensor(FeatureNP).type(torch.float)
MyData = CustomDataSet(features,labels)

# Create Training Data Set
TrainDataloader = DataLoader(MyData, batch_size = 64, shuffle=True)

# Model Hyperpparameters
input_size = DataNN.shape[1] - 1 # Minus 1 because one of the columns is target
num_classes = 1 # we only want 1 NN output (edhrec_rank)
learning_rate = 0.0005
batch_size = 64
num_epochs = 50
weight_decay = 0.01
#device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Network
model = NN(input_size = input_size, num_classes= num_classes).to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr= learning_rate, weight_decay = weight_decay)

# Train Network
for epoch in range(num_epochs):
    print(epoch)
    for batch_idx, (data, targets) in enumerate(TrainDataloader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # Data Shape Check
        data = data.reshape(data.shape[0],-1)
        # forward
        scores = model(data)
        scores = scores.reshape(targets.size())
        loss= criterion(scores,targets)
        
        # backwards
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent, adamW step
        optimizer.step()
    
    print('Loss = ' ,loss.item())
print("Done")