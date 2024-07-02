# The Purpose of this code snippet is to load the card data from the json file and sort them.
import pandas as pd
import json
import os
import sys
sys.path.append(os.getcwd() + "\\Lib")
import ParseFunctions as ParseFunc

f = open('Scryfall Uploads/AllCards_7_1_2024.json', encoding='utf8')
data = json.load(f)
print("Data Loaded")

PandaData = pd.DataFrame.from_records(data)
Columns = PandaData.columns
# Columns of Interest
CoI = ['name','cmc','type_line','oracel_text','color_identity','keywords','legalities','']

# I'm a Commander Player, so it's the only format that I care about legality wise
rowdrop = []
for index, row in  PandaData.iterrows():
    Legalities = row['legalities']    
    if Legalities['commander'] == 'not_legal':
        rowdrop.append(index)                           # create an array with all non-legal card indices
PandaData = PandaData.drop(rowdrop)                     # Drop all cards that are not commander legal
print("Dropped ",len(rowdrop), " rows")
ParseFunc.WriteJsonFromPD(PandaData,'CommanderLegal')

print("Create Changes")
print("End")