# Overview

An attempt to reproduce the paper titled "Assertion Detection in Clinical Natural Language Processing: A Knowledge-Poor Machine Learning Approach".

The data for this study is available from the Department of Biomedical Informatics @ Harvard University. You have to request access to this data.

## Data Extraction

### Negex Data Extraction

You can use the data extraction script in negex_data/extract_data.py to extract data from the Negex project (https://code.google.com/archive/p/negex/)

### NLP 2010 Relations Challenge

You can download data from the 2010 challenge as part of the n2c2 dataset and use the following script to extract the data into usable format:
`./train/extract_data.py`

This script works on the training and test data for the challenge.

## Training & Evaluating the model

The training and evaluation code is all located in main.py. You must first install the dependencies
`pipenv install`

After the dependencies have been installed, you can run `python3 main.py` and the model will run the training and evaluation and output results after each epoch and as part of the evaluation.

## Reproduction Results

| Label        | F1 Score |
| ------------ | -------- |
| Present      | 0.752    |
| Absent       | 0.521    |
| Hypothetical | 0.51     |
| Possible     | 0.49     |
| Conditional  | 0.44     |
| AWSE         | 0.25     |
| Average      | 0.494    |
