import pandas as pd
import json
import os
import sys
import numpy as np
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
sys.path.append(os.getcwd() + "\\Lib")

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer, TrainingArguments, Trainer, pipeline
from datasets import load_dataset, load_metric

# Load PandaDataSet
f = open('Parsed Data Sets/SmallSetPreClassifier.json', encoding='utf8')
data = json.load(f)
ClassifierData = pd.DataFrame.from_records(data)

print(ClassifierData.tail())

# Let's go ahead and reduce this to the columns we actually need for this particular step
CoI = ['edhrec_rank','oracle_text']

for x in ClassifierData.columns:
    if x not in CoI:
        ClassifierData = ClassifierData.drop(x, axis=1)
print(ClassifierData.tail())

# Select Model-base
checkpoint = "distilbert-base-uncased"

# Select our Tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=10)

# Collate Data to speed up training through conversion to PyTorch tensors
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

