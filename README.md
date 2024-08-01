Description of Strategy
- This essentially boils down to a regression problem
    - Current loss function = MAE losss
- Oracle text is currently assigned a label and a score based on a text classification model (Huggingfac BERT Model)
- Inputs include oracle text values, mana cost bin values, card type binary tags.

Current Results:
Model usually has L1Loss errors in the mid teens (15-19%)
Ultimately, this means that the tool is useful in the sense that it's better than strictly guessing, but that 15% difference in edhrec rank can span the difference from being used in every conceivable deck (Sol Ring), to being used in less than 1% of capable decks.

Improvement TODO:
- It doesn't make sense to be comparing cards of disparate types to each other. Reparse data based on card type and then train individualized models for each.
- Text Classification
    - Improvement TODO: Edhrec Rank is not debiased with respect to other training inputs
- Data Normalization
    - Improvement TODO: Current normalization is min max, adjust to z-score normalization (mean = 0, variance = 1)