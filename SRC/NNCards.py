# The Purpose of this code snippet is to load the card data from the json file and sort them.
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
import torchvision.datasets as datasets         # Gives us access to datasets that already exist
import torchvision.transforms as transforms     # Gives us access to transformations on data set

# Pre Network Data Manipulation
import pandas as pd
import numpy as np
import ParseFunctions as PF
import sklearn.preprocessing as SKPRE

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,num_classes)
        
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

# Check that Graphics Card is Available

# Load PandaDataSet
f = open('Parsed Data Sets/FullPostOracleAnalysis.json', encoding='utf8')
data = json.load(f)
Data = pd.DataFrame.from_records(data)
DataNN = Data
DataColumns = Data.columns
ColumnsToKeep = ['cmc','edhrec_rank', 'power',
       'toughness','Creature', 'Enchantment', 'Sorcery',
       'Instant', 'Artifact', 'Battle', 'Planeswalker', '{W}', '{U}', '{B}',
       '{R}', '{G}', '{C}', '{W/P}', '{U/P}', '{B/P}', '{R/P}', '{G/P}',
       '{2/W}', '{2/U}', '{2/B}', '{2/R}', '{2/G}', '{C/W}', '{C/U}', '{C/B}',
       '{C/R}', '{C/G}', '{W/U}', '{W/B}', '{B/R}', '{B/G}', '{U/B}', '{U/R}',
       '{R/G}', '{R/W}', '{G/W}', '{G/U}', '{W/U/P}', '{W/B/P}', '{B/R/P}',
       '{B/G/P}', '{U/B/P}', '{U/R/P}', '{R/G/P}', '{R/W/P}', '{G/W/P}',
       '{G/U/P}', 'LABEL_0', 'LABEL_1', 'LABEL_2', 'score']

for Column in DataColumns:
    if Column not in ColumnsToKeep:
        DataNN = DataNN.drop(Column,axis=1)

# Fix Power and Toughness Values
DataNN['toughness'] = DataNN['toughness'].apply(PF.PowToughFix)
DataNN['power'] = DataNN['power'].apply(PF.PowToughFix)
DataNNCol = DataNN.columns

# Scale All values in DataNN
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
input_size = 55
num_classes = 1 # we only want 1 NN output (edhrec_rank)
learning_rate = 0.001
batch_size = 64
num_epochs = 1
#device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Network
model = NN(input_size = input_size, num_classes= num_classes).to(device)

# Loss and Optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

# Train Network
for epoch in range(num_epochs):
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
        
        # gradient descent vs adam step
        optimizer.step()

# Check Accuracy
def check_accuracy(loader,model):
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)
            
            scores = model(x)
            