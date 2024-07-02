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
