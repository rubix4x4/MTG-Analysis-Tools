# The Purpose of this code is to prepare data for Text Classification Training.
import pandas as pd
import json
import os
import sys
import numpy as np
sys.path.append(os.getcwd() + "\\Lib")
import ParseFunctions as PF


f = open('Parsed Data Sets/CommanderLegal.json', encoding='utf8')
data = json.load(f)
PandaData = pd.DataFrame.from_records(data)

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
print("EDHREC Rank Normalized")
PandaData['cmc'] = PandaData['cmc'].apply(cmcNormalize)
print("Overall CmC Normalized")

# Create Card Type Binary Values
PandaData = PF.TagCardTypes(PandaData)

# Create Reinterpreted Mana Cost data





print(PandaData.tail(n=20))

