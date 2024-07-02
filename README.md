Initial Readme file

THE PLAN (Work in progress)

Data normalization cleaning
- Normalize EDHREC rank values
- Create Binary Value to represent card types
- Create Binary Value to represent if a card transforms


Text Classification Step
- remove keyword explanations from oracle text
- combine oracletext of cards with multiple side/modes
- Create 3 Labels for cards ("In over 10%","5 to 10%, and 1 to 5%, less than 1%)
    - ID EDHREC_Rank Cutoffs for these bins
- Feed Classifier bin and confidence

PyTorch
- Feed data frame values
- Feed Classifier Bin and Confidence Score
