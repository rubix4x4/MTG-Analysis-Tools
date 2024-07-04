import numpy as np
import json
import pandas as pd
import re

def WriteJsonFromPD(Data,FileName):
    result = Data.to_json(orient="records")
    parsed = json.loads(result)
    s = json.dumps(parsed, indent = 4)
    FileLocation = "Parsed Data Sets"
    FileOutType = ".json"
    FileOut = FileLocation + "/" + FileName + FileOutType
    open(FileOut,"w").write(s)

def TagCardTypes(Data):
    CardTypes = ['Creature','Enchantment','Sorcery','Instant','Artifact','Battle', 'Planeswalker']
    CardListArray = []
    for index, row in  Data.iterrows():
        CardTypeVector = []
        for type in CardTypes:
            if type in row['type_line']:
                CardTypeVector.append(1)
            else:
                CardTypeVector.append(0)
        CardListArray.append(CardTypeVector)
    
    # Turn Card List Array into a dataframe object
    CardTypeDF = pd.DataFrame(CardListArray, columns = CardTypes)
    NewSet = Data.join(CardTypeDF)
    return NewSet

def TagManaCosts(Data):
    ManaCost_Tags = ['{W}','{U}','{B}','{R}','{G}', '{C}'                                                           # WUBRG
        ,'{W/P}','{U/P}','{B/P}','{R/P}','{G/P}'                                                                    # Phyrexian WUBRG
        ,'{2/W}','{2/U}','{2/B}','{2/R}','{2/G}'                                                                    # 2 or Color
        ,'{C/W}','{C/U}','{C/B}','{C/R}','{C/G}'                                                                    # Colorless or Color
        ,'{W/U}','{W/B}','{B/R}','{B/G}','{U/B}','{U/R}','{R/G}','{R/W}','{G/W}','{G/U}'                            # Guild Color
        ,'{W/U/P}','{W/B/P}','{B/R/P}','{B/G/P}','{U/B/P}','{U/R/P}','{R/G/P}','{R/W/P}','{G/W/P}','{G/U/P}']       # Phyrexian Guild Color
    ManaArray = []
    for index, row in  Data.iterrows():
        CardManaVector = []
        ManaCost = row['mana_cost']
        # Special Cases: when card has multiple faces, sum the two cards together
        if row['card_faces'] != None:                                                   
                ManaCost = ""
                for card in row['card_faces']:
                    ManaCost = ManaCost + ' ' + card['mana_cost']
        # Seach ManaCost for different tags
        for Tag in ManaCost_Tags:
            if Tag in ManaCost:
                    CardManaVector.append(ManaCost.count(Tag))
            else:
                CardManaVector.append(0)
        ManaArray.append(CardManaVector)
    
    # Turn Mana Array into a dataframe object and join to Data
    ManaCostDF = pd.DataFrame(ManaArray, columns = ManaCost_Tags)
    NewSet = Data.join(ManaCostDF)
    return NewSet

def OracleCleanup(Data):
    for index, card in  Data.iterrows():
        # Concatenate cards with multiple faces
        Names = []           # Initialize Empty Name
        OText = ""          # Initialize Empty String OracleText
        if card['card_faces'] != None:
            for CardFace in card['card_faces']:
                Names.append(CardFace['name'])
                OText = OText + CardFace['oracle_text']
        else:
            Names.append(card['name'])
            OText = card['oracle_text']
        OText = re.sub("[\(\[].*?[\)\]]", "", OText)
        
        for Name in Names:
            OText = re.sub(re.escape(Name),"Self",OText)
     
        OText = re.sub("\n"," ",OText)    
        # Replace Uncleaned Oracle Text
        Data.loc[index, 'oracle_text'] = OText   
    return Data
        
def EdhrecLabels(Data):
    Labels = []
    for index, card in  Data.iterrows():
        Edhrec_Rank = card['edhrec_rank']
        
        if Edhrec_Rank >= 0.11746066433138619:
            Labels.append(2) # Less than 2%
        
        elif (Edhrec_Rank >= 0.055179195802186075 and Edhrec_Rank <0.11746066433138619):
            Labels.append(1) # Between 2 to 5%
            
        elif Edhrec_Rank < 0.055179195802186075:
            Labels.append(0) # Greater than 5%
    return Labels
