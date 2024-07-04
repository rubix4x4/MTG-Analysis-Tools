Initial Readme file

THE PLAN (Work in progress)

Data normalization cleaning
- Normalize EDHREC rank values
- Create Binary Value to represent card types
- Create Binary Value to represent if a card transforms


Text Classification Step
- remove keyword explanations from oracle text
- combine oracletext of cards with multiple side/modes
- Create 3 Labels for cards ("In over 5%", 2 to 5%, less than 2%)

    Limitation of Current Classifier:
        EDHREC_RANK is affected by more than just oracle text when assigning labels
        Therefore, future improvements should look at removing effect of other card factors
        before assigning labels for classification

PyTorch
- Feed data frame values
- Feed Classifier Bin and Confidence Score
