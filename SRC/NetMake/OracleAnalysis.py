# This section of code takes the card data and populates label and confidence score fields for each card based on their oracle text
import os
import sys
sys.path.append(os.getcwd() + "\\Lib")
import ParseFunctions as PF

import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

import time

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

#model_name = "Test_Train_Location"
model_name = "Models\MTG_Oracle_Analysis_Creature"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline ("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

datapandas = pd.read_json('Parsed Data Sets/PreClass/CreaturePreClass.json')
#datapandas = pd.read_json('Parsed Data Sets/SmallSetPreClassifier.json')

TierLabels = ['LABEL_0','LABEL_1','LABEL_2']
TierVector = []
TierArray = []
ConfidenceArray = []

# Check if can do all at once?
start = time.time()
print("Start Oracle Analysis")

JustText = datapandas['oracle_text']
TextList = JustText.to_list()
OracleAnalyzed = classifier(TextList)

end = time.time()
Duration = end-start
print("Oracle Analysis End after", Duration, "seconds")

print("Write to DataFrame)")
for Result in OracleAnalyzed:
    TierVector = []
    Tier = Result['label']
    Confidence = Result['score']
    # Creates an vector representing which tier the card is in
    for TierName in TierLabels:
        if Tier == TierName:
            TierVector.append(1)
        else:
            TierVector.append(0)
    
    # Confidence Score
    TierArray.append(TierVector)
    ConfidenceArray.append(Confidence)

# Create a pandas dataset from Tier Array and confidence Array, then append to datapandas
TierDF = pd.DataFrame(TierArray, columns = TierLabels)
ConfidenceDF = pd.DataFrame(ConfidenceArray, columns = ['score'])
datapandas = datapandas.join(TierDF)
datapandas = datapandas.join(ConfidenceDF)
PF.WriteJsonFromPD(datapandas,'CreaturePostOracle')
print("End")