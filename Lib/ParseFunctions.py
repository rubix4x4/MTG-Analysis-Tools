import json
import pandas as pd

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
    print ("End")
    NewSet = Data.join(CardTypeDF)
    return NewSet
        