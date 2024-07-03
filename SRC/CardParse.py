# The Purpose of this code snippet is to load the card data from the json file and sort them.
import pandas as pd
import json
import os
import sys
import numpy as np
sys.path.append(os.getcwd() + "\\Lib")
import ParseFunctions as ParseFunc

f = open('Scryfall Uploads/AllCards_7_1_2024.json', encoding='utf8')
data = json.load(f)
print("Data Loaded")

PandaData = pd.DataFrame.from_records(data)
Columns = PandaData.columns
# Columns of Interest
CoI = ['name','cmc', 'mana_cost','type_line','oracel_text','color_identity','keywords','legalities','power','toughness','edhrec_rank']

for x in Columns:
    if x not in CoI:
        PandaData = PandaData.drop(x, axis=1)

# I'm a Commander Player, so it's the only format that I care about legality wise
rowdrop = []
for index, row in  PandaData.iterrows():
    Legalities = row['legalities']    
    if Legalities['commander'] == 'not_legal':
        rowdrop.append(index)                           # create an array with all non-legal card indices
PandaData = PandaData.drop(rowdrop)                     # Drop all cards that are not commander legal
print("Dropped ",len(rowdrop), " rows")
#ParseFunc.WriteJsonFromPD(PandaData,'CommanderLegal')


# Normalize card value
WorstCardValue = max(PandaData[:]['edhrec_rank'])       # I'll be using edhrec rank as a objective measure of a cards approximate value/power
def EDHRECNormalize(a):
    if np.isnan(a):
        a = WorstCardValue
    Value = a/(WorstCardValue+1e-6)
    return Value
def cmcNormalize(a):
    Value = a/max(PandaData['cmc'])
    return Value
PandaData['edhrec_rank'] = PandaData['edhrec_rank'].apply(EDHRECNormalize)
PandaData['cmc'] = PandaData['cmc'].apply(cmcNormalize)






print("End")