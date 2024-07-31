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
    Reg_Cost = ['{W}','{U}','{B}','{R}','{G}', '{C}']                                                                       # WUBRGC
    RegOrTwo = ['{2/W}','{2/U}','{2/B}','{2/R}','{2/G}']                                                                    # 2 or Color
    OptionalColor = ['{C/W}','{C/U}','{C/B}','{C/R}','{C/G}']                                                                    # Colorless or Color
    GuildColor = ['{W/U}','{W/B}','{B/R}','{B/G}','{U/B}','{U/R}','{R/G}','{R/W}','{G/W}','{G/U}']                           # Guild Color
    ManaArray = [0,0,0,0,0,0]
    CardManaVector = []
    # This map correlates the Tag to how each count thereof adjust the ManaArray
    GuildMap = {'{W/U}':[0.5, 0.5, 0, 0, 0, 0],
        '{W/B}':[0.5, 0, 0.5, 0, 0, 0,],
        '{B/R}':[0, 0, 0.5, 0.5, 0, 0],
        '{B/G}':[0, 0, 0.5, 0, 0.5, 0],
        '{U/B}':[0, 0.5, 0.5, 0, 0, 0],
        '{U/R}':[0, 0.5, 0, 0.5, 0, 0],
        '{R/G}':[0, 0, 0, 0.5, 0.5, 0],
        '{R/W}':[0.5, 0, 0, 0.5, 0, 0],
        '{G/W}':[0.5, 0, 0, 0, 0.5, 0],
        '{G/U}':[0, 0.5, 0, 0, 0.5, 0],
        '{C/W}':[0.5, 0, 0, 0, 0, 0.5],
        '{C/U}':[0, 0.5, 0, 0, 0, 0.5],
        '{C/B}':[0, 0, 0.5, 0, 0, 0.5],
        '{C/R}':[0, 0, 0, 0.5, 0, 0.5],
        '{C/G}':[0, 0, 0, 0, 0.5, 0.5]
        }
      
    for index, row in  Data.iterrows():
        # Adjust CardManaVector to just WURBG and Colorless
        # Phyrexian Mana is counted as free as it functionally has no cost in commander
        ManaArray = [0,0,0,0,0,0]
        ManaCost = row['mana_cost']
        # Special Cases: when card has multiple faces, sum the two cards together
        if row['card_faces'] != None:                                                   
            ManaCost = ""
            for card in row['card_faces']:
                ManaCost = ManaCost + ' ' + card['mana_cost']
        
        # Seach ManaCost for Regular Costs
        for Ind, Tag in enumerate(Reg_Cost):
            Count = ManaCost.count(Tag)
            ManaArray[Ind] += Count
        
        # Seach ManaCost for RegOrTwo
        for Ind, Tag in enumerate(RegOrTwo):
            Count = ManaCost.count(Tag)
            ManaArray[Ind] += Count # Adds number of counts. Most will try to use colored mana over paying 2 of generic

        # Split the difference between colored and the colorless cost
        NPManaArray = np.array(ManaArray, dtype=float)
        for Tag in OptionalColor:
            OptionalColorArray = GuildMap[Tag]
            NPOptionalColorArray = np.array(OptionalColorArray)
            Count = ManaCost.count(Tag)
            ManaAdjust = Count*NPOptionalColorArray
            NPManaArray += ManaAdjust
        
        # Guild Colors
        # Convert ManaArray to an np array for next step
        for Tag in GuildColor:
            GuildArray = GuildMap[Tag]
            NPGuildArray = np.array(GuildArray)
            Count = ManaCost.count(Tag)
            ManaAdjust = Count*NPGuildArray
            NPManaArray += ManaAdjust
            
            
        # NOTE PHYREXIAN Mana has been ignored. For the purposes of commander, 
        # I am making the assumption that most players will choose to pay 2 life over the mana cost
            
        ManaArray = NPManaArray.tolist()
        CardManaVector.append(ManaArray)

    # Turn Mana Array into a dataframe object and join to Data
    ManaCostDF = pd.DataFrame(CardManaVector, columns = Reg_Cost)
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

def PowToughFix(value):
    if value == None:
        value = 0
    elif '*' in str(value):
        value = re.sub(re.escape('*'),'0',value)
        value = eval(value)
    else:
        value = eval(value)
    return value