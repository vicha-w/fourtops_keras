# fourtops_keras

## What you need
* **Python 3** (Code in this repository is written for Python 3 only. Sorry for that.)
* keras
* pandas
* root_numpy (For interfaces with ROOT file)
* pyROOT (For interfaces with certain ROOT classes - used in Toolset.py)
* matplotlib

## Contents
`Toolset.py` is a support library with many functions, such as converting Craneen files into pandas dataframe, plotting ROC curves, calculate overlapping coefficients, and calculate signal_acceptance

`root2pandas.py` is a Python script that generates pandas dataframe containing all variables used in ML training with Keras neural nets. All information of extra variables not available in original Craneen files can be found in `add_variables` function.
