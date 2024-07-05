import pandas as pd
import json
import os
import sys
import numpy as np
import torch
sys.path.append(os.getcwd() + "\\Lib")
import ParseFunctions as ParseFunc

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer, TrainingArguments, Trainer, pipeline
import evaluate

# region TextClassifier Pretrained Model Definition
# Select Model-base
checkpoint = "distilbert-base-uncased"

# Select our Tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

# Collate Data to speed up training through conversion to PyTorch tensors
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# endregion

# Load PandaDataSet
f = open('Parsed Data Sets/FullSetPreClassifier.json', encoding='utf8')
data = json.load(f)
ClassifierData = pd.DataFrame.from_records(data)

# Let's go ahead and reduce this to the columns we actually need for this particular step
CoI = ['edhrec_rank','oracle_text']

for x in ClassifierData.columns:
    if x not in CoI:
        ClassifierData = ClassifierData.drop(x, axis=1)

# Assign EDHREC Rank Based Labels
Labels = ParseFunc.EdhrecLabels(ClassifierData)
LabelsDF = pd.DataFrame(Labels, columns = ['labels'])
ClassifierData = ClassifierData.join(LabelsDF)

# Drop Edhrec_Rank from dataset
ClassifierData = ClassifierData.drop('edhrec_rank',axis = 1)

# Convert to Dataset
dataset = Dataset.from_pandas(ClassifierData)
dataset = dataset.rename_column('oracle_text','text') # HuggingFace classifier strictly requires only 'label' and 'text' columns
# Split into Test and Train set
dataset = dataset.train_test_split(test_size = 0.2) # 20% of entries are sequestered to training set
# Tokenize Datasets
def tokenize_function(examples):
   return tokenizer(examples["text"], truncation=True)

tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)


# Define our computed metrics
def compute_metrics(eval_pred):
   load_accuracy = evaluate.load("accuracy")
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   return {"accuracy": accuracy}

# Define our training arguments
training_args = TrainingArguments(
   output_dir='Models\MTG_O_A_Training_Args',
   learning_rate=1e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=4,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=False,
)
 
# Define our trainer class
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())
trainer.save_model('Models\MTG_Oracle_Analysis')

print('checkpoint')