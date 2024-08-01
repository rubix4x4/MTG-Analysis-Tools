# The Purpose of this code is to prepare data for Text Classification Training.
# Data is output to new json file "____PreClassifier.json"
import pandas as pd
import json
import os
import sys
import numpy as np
import matplotlib as plt
sys.path.append(os.getcwd() + "\\Lib")
import ParseFunctions as PF

f = open('Parsed Data Sets/CommanderLegal.json', encoding='utf8')
#f = open('Parsed Data Sets/SmallDevSet.json', encoding='utf8')

data = json.load(f)
PandaData = pd.DataFrame.from_records(data)

# Normalize card value
WorstCardValue = max(PandaData[:]['edhrec_rank'])       # I'll be using edhrec rank as a objective measure of a cards approximate value/power

# Simpler Normalization Functions
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

PandaData = PF.TagCardTypes(PandaData)                  # Tag Card Types and Append New Columns
print("Card Types Tagged")

PandaData = PF.TagManaCosts(PandaData)                  # Tag Card Costs that have special stipulations
print("Mana Costs Tagged")

PandaData = PF.OracleCleanup(PandaData)                 # Replace cardname text with Self, remove keyword explanations
print("Oracle text Cleanup")

#PF.WriteJsonFromPD(PandaData.tail(n=5000),'SmallSetPreClassifier')
PF.WriteJsonFromPD(PandaData,'FullSetPreClassifier')
PF.WriteJsonFromPD(PandaData[(PandaData['Creature'] == 1)],'CreaturePreClass')
PF.WriteJsonFromPD(PandaData[(PandaData['Artifact'] == 1)],'ArtifactPreClass')
PF.WriteJsonFromPD(PandaData[(PandaData['Sorcery'] == 1)],'SorceryPreClass')
PF.WriteJsonFromPD(PandaData[(PandaData['Instant'] == 1)],'InstantPreClass')
PF.WriteJsonFromPD(PandaData[(PandaData['Planeswalker'] == 1)],'PlanesPreClass')
PF.WriteJsonFromPD(PandaData[(PandaData['Battle'] == 1)],'BattlePreClass')
print(PandaData.tail(n=5))

