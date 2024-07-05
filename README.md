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
- Feed Classifier Bin Tag and Confidence Score
- 2 hidden layers with 55 nodes each (same size as input layer)
- relu between each layer
- L1Loss

Current Results:
Model usually has L1Loss errors in the mid teens (15-19%)
Ultimately, this means that the tool is useful in the sense that it's better than strictly guessing, but that 15% difference in edhrec rank can span the difference from being used in every conceivable deck (Sol Ring), to being used in less than 1% of capable decks.

Still, this was a good foray into learning basic machine learning and pytorch framework. I may revisit this tool in the future but for now it will stick as is until I can get external expert opinions on how to improve the process.
