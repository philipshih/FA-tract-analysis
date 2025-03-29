# Tract-Specific AFQ Analyses
# Philip Shih

# Section I: Setup
### Import libraries.


```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf  
from statsmodels.formula.api import ols
from scipy.stats import shapiro, skew, kurtosis
import scipy.stats as stats
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, lasso_path
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import random

```


```python
current_path = os.getcwd()
print(f"Current Working Directory: {current_path}")
```

    Current Working Directory: /Users/kjs/Library/CloudStorage/Box-Box/Medical_School/AFQ_Analysis/current_analyses


### Extract all FA data from .csv files generated from AFQ 'profiles' method into a pandas dataframe.

```python
# Define the relative path to the directory containing the CSV files
directory = os.path.join('..', 'data', 'raw', 'mrtrix_csvs')

# Regex to match the filenames
pattern = r'sub-PDa\d{3}_ses-01_dwi_space-RASMM_model-probCSD_algo-reco80_desc-profiles_dwi.csv'

# List all files in 'directory' and filter based on the pattern
afq_files = [os.path.join(directory, f) for f in os.listdir(directory) if re.match(pattern, f)]

fa_data_list = []

# Loop through the file list and read each file into a dataframe
for file in afq_files:
    df = pd.read_csv(file)
    # Extract the subject ID from the file name using os.path.basename for cross-platform compatibility
    subject_id = os.path.basename(file).split('_')[0]
    # Add a column with the subject ID
    df['subject_id'] = subject_id
    # Append the dataframe to the list
    fa_data_list.append(df)

# Combine all dataframes into one
combined_fa_data = pd.concat(fa_data_list, ignore_index=True)

# Check the first few rows of the combined dataframe
combined_fa_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nodeID</th>
      <th>dti_fa</th>
      <th>tractID</th>
      <th>subject_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.319093</td>
      <td>ATR_L_fa</td>
      <td>sub-PDa375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.176145</td>
      <td>ATR_L_fa</td>
      <td>sub-PDa375</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.282691</td>
      <td>ATR_L_fa</td>
      <td>sub-PDa375</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.352052</td>
      <td>ATR_L_fa</td>
      <td>sub-PDa375</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.351703</td>
      <td>ATR_L_fa</td>
      <td>sub-PDa375</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>289995</th>
      <td>96</td>
      <td>0.397247</td>
      <td>UF_R_fa</td>
      <td>sub-PDa096</td>
    </tr>
    <tr>
      <th>289996</th>
      <td>97</td>
      <td>0.185224</td>
      <td>UF_R_fa</td>
      <td>sub-PDa096</td>
    </tr>
    <tr>
      <th>289997</th>
      <td>98</td>
      <td>0.232283</td>
      <td>UF_R_fa</td>
      <td>sub-PDa096</td>
    </tr>
    <tr>
      <th>289998</th>
      <td>99</td>
      <td>0.308811</td>
      <td>UF_R_fa</td>
      <td>sub-PDa096</td>
    </tr>
    <tr>
      <th>289999</th>
      <td>100</td>
      <td>0.411588</td>
      <td>UF_R_fa</td>
      <td>sub-PDa096</td>
    </tr>
  </tbody>
</table>
<p>290000 rows × 4 columns</p>
</div>



### Load patient outcome measures.

### Prepare outcome measures for merge with FA data by standardizing column names.

Set desired timepoint column name to delta_ledd1. Ex: 'LEDDpost4_Delta': 'delta_ledd1'


```python
# Define the relative path to the outcome measures data file
outcome_file = os.path.join('..', 'data', 'metadata', 'Deidentified_master_spreadsheet_01.07.25_SD.xlsx')
outcome_data = pd.read_excel(outcome_file)
outcome_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PTID_Retro_Clin</th>
      <th>PTID_R01_Preop</th>
      <th>PTID_3T_Postop</th>
      <th>PTID_Preop_fMRI_CONN</th>
      <th>Consent_General</th>
      <th>Consent_R01</th>
      <th>R01_Prelim_Analysis</th>
      <th>Preop_R01_Candidate</th>
      <th>Postop_3T_Candidate</th>
      <th>Postop _15T_Candidate</th>
      <th>...</th>
      <th>R01_Beck Anxiety Inventory (BAI): Raw</th>
      <th>R01_Beck Anxiety Inventory (BAI): T-Score</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): Raw</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): T-Score</th>
      <th>R01_Starkstein Apathy Scale (SAS): Raw</th>
      <th>R01_Starkstein Apathy Scale (SAS): T-Score</th>
      <th>R01_Non-Motor Symptoms Questionnaire (NMS-Quest): Raw</th>
      <th>R01_QUIP-A</th>
      <th>R01_Columbia Suicide Severity Rating Scale (C-SSRS): Raw</th>
      <th>R01_Neuropsychology_Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PDa001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PDa002</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-02-01 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PDa003</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-02-29 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PDa004</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-03-14 00:00:00</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PDa005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-03-15 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 306 columns</p>
</div>




```python
    outcome_data = outcome_data.rename(columns={'PTID_Retro_Clin': 'subject_id', 'LEDDpost1_Delta': 'delta_ledd1', 'LEDDpost2_Delta': 'delta_ledd2', 'LEDDpost3_Delta': 'delta_ledd3'})

# Ensure the subject_id format matches the one used in FA data
outcome_data['subject_id'] = outcome_data['subject_id'].apply(lambda x: f"sub-{x}")
outcome_data = outcome_data.drop_duplicates()
outcome_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>PTID_R01_Preop</th>
      <th>PTID_3T_Postop</th>
      <th>PTID_Preop_fMRI_CONN</th>
      <th>Consent_General</th>
      <th>Consent_R01</th>
      <th>R01_Prelim_Analysis</th>
      <th>Preop_R01_Candidate</th>
      <th>Postop_3T_Candidate</th>
      <th>Postop _15T_Candidate</th>
      <th>...</th>
      <th>R01_Beck Anxiety Inventory (BAI): Raw</th>
      <th>R01_Beck Anxiety Inventory (BAI): T-Score</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): Raw</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): T-Score</th>
      <th>R01_Starkstein Apathy Scale (SAS): Raw</th>
      <th>R01_Starkstein Apathy Scale (SAS): T-Score</th>
      <th>R01_Non-Motor Symptoms Questionnaire (NMS-Quest): Raw</th>
      <th>R01_QUIP-A</th>
      <th>R01_Columbia Suicide Severity Rating Scale (C-SSRS): Raw</th>
      <th>R01_Neuropsychology_Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-PDa001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-PDa002</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-02-01 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-PDa003</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-02-29 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-PDa004</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-03-14 00:00:00</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-PDa005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-03-15 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>532</th>
      <td>sub-nan</td>
      <td>NaN</td>
      <td>KEY B-D</td>
      <td>NaN</td>
      <td>KEY ALL</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>533</th>
      <td>sub-nan</td>
      <td>NaN</td>
      <td>Eligible for study, \nbut not yet in the study</td>
      <td>NaN</td>
      <td>Chart  section complete</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>534</th>
      <td>sub-nan</td>
      <td>NaN</td>
      <td>Determining if eligible for study</td>
      <td>NaN</td>
      <td>Chart section incomplete</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>535</th>
      <td>sub-nan</td>
      <td>NaN</td>
      <td>NA = not eligible for study</td>
      <td>NaN</td>
      <td>Cant use patients data</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>536</th>
      <td>sub-nan</td>
      <td>NaN</td>
      <td>PTID= in study and/or \nmaster database</td>
      <td>NaN</td>
      <td>Need to update chart imaging</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>537 rows × 306 columns</p>
</div>



### Create 'data' dataframe. 
 Each row of 'data' represents a node of a patient's white matter tract and includes the FA value of that node, together with the patient's corresponding outcome measures.


```python
# Merge the FA data with the outcome measure data on the subject_id
data = pd.merge(combined_fa_data, outcome_data)

# Display the first few rows of the merged dataframe to verify the merge
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nodeID</th>
      <th>dti_fa</th>
      <th>tractID</th>
      <th>subject_id</th>
      <th>PTID_R01_Preop</th>
      <th>PTID_3T_Postop</th>
      <th>PTID_Preop_fMRI_CONN</th>
      <th>Consent_General</th>
      <th>Consent_R01</th>
      <th>R01_Prelim_Analysis</th>
      <th>...</th>
      <th>R01_Beck Anxiety Inventory (BAI): Raw</th>
      <th>R01_Beck Anxiety Inventory (BAI): T-Score</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): Raw</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): T-Score</th>
      <th>R01_Starkstein Apathy Scale (SAS): Raw</th>
      <th>R01_Starkstein Apathy Scale (SAS): T-Score</th>
      <th>R01_Non-Motor Symptoms Questionnaire (NMS-Quest): Raw</th>
      <th>R01_QUIP-A</th>
      <th>R01_Columbia Suicide Severity Rating Scale (C-SSRS): Raw</th>
      <th>R01_Neuropsychology_Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.319093</td>
      <td>ATR_L_fa</td>
      <td>sub-PDa375</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>72</td>
      <td>2022-11-07 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.176145</td>
      <td>ATR_L_fa</td>
      <td>sub-PDa375</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>72</td>
      <td>2022-11-07 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.282691</td>
      <td>ATR_L_fa</td>
      <td>sub-PDa375</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>72</td>
      <td>2022-11-07 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.352052</td>
      <td>ATR_L_fa</td>
      <td>sub-PDa375</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>72</td>
      <td>2022-11-07 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.351703</td>
      <td>ATR_L_fa</td>
      <td>sub-PDa375</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>72</td>
      <td>2022-11-07 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>289995</th>
      <td>96</td>
      <td>0.397247</td>
      <td>UF_R_fa</td>
      <td>sub-PDa096</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24</td>
      <td>2020-10-12 00:00:00</td>
      <td>NaT</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>289996</th>
      <td>97</td>
      <td>0.185224</td>
      <td>UF_R_fa</td>
      <td>sub-PDa096</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24</td>
      <td>2020-10-12 00:00:00</td>
      <td>NaT</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>289997</th>
      <td>98</td>
      <td>0.232283</td>
      <td>UF_R_fa</td>
      <td>sub-PDa096</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24</td>
      <td>2020-10-12 00:00:00</td>
      <td>NaT</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>289998</th>
      <td>99</td>
      <td>0.308811</td>
      <td>UF_R_fa</td>
      <td>sub-PDa096</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24</td>
      <td>2020-10-12 00:00:00</td>
      <td>NaT</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>289999</th>
      <td>100</td>
      <td>0.411588</td>
      <td>UF_R_fa</td>
      <td>sub-PDa096</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24</td>
      <td>2020-10-12 00:00:00</td>
      <td>NaT</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>290000 rows × 309 columns</p>
</div>




```python
columns_to_check = ['subject_id', 'delta_ledd1', 'MDSUPDRSIIIpre_Percent_TOTAL_V']

# Select the specified columns from 'outcome_data'
df_selected = data[columns_to_check]

# Create a new DataFrame for the output
new_df = df_selected.copy()

# For the specified columns, mark '1' if value is present, '0' if missing
for col in columns_to_check[1:]:  # Skip 'PTID_Retro_Clin'
    new_df[col] = df_selected[col].notna().astype(int)

# Output the new DataFrame to a CSV file in the results directory
df_cleaned = new_df.drop_duplicates(subset='subject_id').sort_values(by='subject_id')
results_dir = os.path.join('..', 'results')
os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists
df_cleaned.to_csv(os.path.join(results_dir, 'missing_values.csv'), index=True)
# Display the new DataFrame (optional)
print(df_cleaned)
```

            subject_id  delta_ledd1  MDSUPDRSIIIpre_Percent_TOTAL_V
    166900  sub-PDa001            1                               1
    93300   sub-PDa002            1                               0
    34000   sub-PDa003            1                               0
    57300   sub-PDa005            1                               1
    268500  sub-PDa006            1                               1
    ...            ...          ...                             ...
    242500  sub-PDa399            1                               1
    214500  sub-PDa401            1                               0
    119300  sub-PDa402            1                               1
    89300   sub-PDa409            1                               1
    12000   sub-PDa411            1                               1
    
    [146 rows x 3 columns]


# Section II: Tract-Specific Analysis

The following contains tract-specific FA values with their corresponding patient data.

### Merge nodes of 'data' into tracts, averaging their FA values.


```python
# Group data by subject and tract to calculate average FA value for each
average_fa_per_subject_tract = data.groupby(['subject_id', 'tractID'])['dti_fa'].mean().reset_index()
average_fa_per_subject_tract
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>tractID</th>
      <th>dti_fa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-PDa001</td>
      <td>ATR_L_fa</td>
      <td>0.275715</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-PDa001</td>
      <td>ATR_R_fa</td>
      <td>0.304207</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-PDa001</td>
      <td>CGC_L_fa</td>
      <td>0.244211</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-PDa001</td>
      <td>CGC_R_fa</td>
      <td>0.230977</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-PDa001</td>
      <td>CGH_L_fa</td>
      <td>0.208619</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2895</th>
      <td>sub-PDa411</td>
      <td>SLFT_R_fa</td>
      <td>0.417561</td>
    </tr>
    <tr>
      <th>2896</th>
      <td>sub-PDa411</td>
      <td>SLF_L_fa</td>
      <td>0.274110</td>
    </tr>
    <tr>
      <th>2897</th>
      <td>sub-PDa411</td>
      <td>SLF_R_fa</td>
      <td>0.280753</td>
    </tr>
    <tr>
      <th>2898</th>
      <td>sub-PDa411</td>
      <td>UF_L_fa</td>
      <td>0.188834</td>
    </tr>
    <tr>
      <th>2899</th>
      <td>sub-PDa411</td>
      <td>UF_R_fa</td>
      <td>0.323441</td>
    </tr>
  </tbody>
</table>
<p>2900 rows × 3 columns</p>
</div>



### Create 'tract_specific_data'.
Each row of 'tract_specific_data' represents white matter tracts that have had the FA value of their nodes averaged into one value.


```python
tract_specific_data = pd.merge(average_fa_per_subject_tract, outcome_data, on='subject_id', how='inner')
```

### Fix inconsistent spreadsheet data typing for delta_ledd1 and delta_ledd0.
Ex: "58.1" -> 58.1


```python
tract_specific_data['delta_ledd1'] = pd.to_numeric(tract_specific_data['delta_ledd1'], errors='coerce') 
tract_specific_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>tractID</th>
      <th>dti_fa</th>
      <th>PTID_R01_Preop</th>
      <th>PTID_3T_Postop</th>
      <th>PTID_Preop_fMRI_CONN</th>
      <th>Consent_General</th>
      <th>Consent_R01</th>
      <th>R01_Prelim_Analysis</th>
      <th>Preop_R01_Candidate</th>
      <th>...</th>
      <th>R01_Beck Anxiety Inventory (BAI): Raw</th>
      <th>R01_Beck Anxiety Inventory (BAI): T-Score</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): Raw</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): T-Score</th>
      <th>R01_Starkstein Apathy Scale (SAS): Raw</th>
      <th>R01_Starkstein Apathy Scale (SAS): T-Score</th>
      <th>R01_Non-Motor Symptoms Questionnaire (NMS-Quest): Raw</th>
      <th>R01_QUIP-A</th>
      <th>R01_Columbia Suicide Severity Rating Scale (C-SSRS): Raw</th>
      <th>R01_Neuropsychology_Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-PDa001</td>
      <td>ATR_L_fa</td>
      <td>0.275715</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-PDa001</td>
      <td>ATR_R_fa</td>
      <td>0.304207</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-PDa001</td>
      <td>CGC_L_fa</td>
      <td>0.244211</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-PDa001</td>
      <td>CGC_R_fa</td>
      <td>0.230977</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-PDa001</td>
      <td>CGH_L_fa</td>
      <td>0.208619</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2895</th>
      <td>sub-PDa411</td>
      <td>SLFT_R_fa</td>
      <td>0.417561</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91</td>
      <td>2023-09-01 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2896</th>
      <td>sub-PDa411</td>
      <td>SLF_L_fa</td>
      <td>0.274110</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91</td>
      <td>2023-09-01 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2897</th>
      <td>sub-PDa411</td>
      <td>SLF_R_fa</td>
      <td>0.280753</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91</td>
      <td>2023-09-01 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2898</th>
      <td>sub-PDa411</td>
      <td>UF_L_fa</td>
      <td>0.188834</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91</td>
      <td>2023-09-01 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2899</th>
      <td>sub-PDa411</td>
      <td>UF_R_fa</td>
      <td>0.323441</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91</td>
      <td>2023-09-01 00:00:00</td>
      <td>NaT</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2900 rows × 308 columns</p>
</div>



### Check for inconsistent spacing of spreadsheet values.

Ex: "GPi " -> "GPi"


```python
for col in tract_specific_data.columns:
    # Check if the column's data type is 'object' (commonly used for strings in pandas)
    if tract_specific_data[col].dtype == 'object':
        # Apply .str.strip() only to string entries
        tract_specific_data[col] = tract_specific_data[col].apply(lambda x: x.strip() if type(x) == str else x)
```

### Output tract-specific CSV data.
```python
results_dir = os.path.join('..', 'results')
os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists
tract_specific_data.to_csv(os.path.join(results_dir, '2024tractspecific.csv'))
tract_specific_data_csv = pd.read_csv(os.path.join(results_dir, '2024tractspecific.csv'))
```

    /var/folders/_t/1k38qqf536x8_n0nbth3c0vm0000gn/T/ipykernel_14277/2208836796.py:2: DtypeWarning: Columns (4,5,6,32,33,38,47,48,49,50,51,52,53,148,152,155,160,168,176,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,215,221,259,283,285,287,289,291,292,293) have mixed types. Specify dtype option on import or set low_memory=False.
      tract_specific_data_csv = pd.read_csv("/Users/kjs/Library/CloudStorage/Box-Box/Medical_School/2024tractspecific.csv")


### Calculate Percentage of Tracts Present


```python
presence_percentage_corrected = tract_specific_data.pivot_table(index='subject_id', columns='tractID', values='dti_fa', aggfunc='first').notna().mean() * 100

# Convert to DataFrame for display
presence_percentage_df = presence_percentage_corrected.reset_index().rename(columns={0: '%_tracts_present'})
results_dir = os.path.join('..', 'results')
os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists
presence_percentage_df.sort_values(by='%_tracts_present', ascending=False).to_csv(os.path.join(results_dir, 'presence_percentage.csv'))
```

### Filter only tracts that are present in most subjects (100/157)

```python
results_dir = os.path.join('..', 'results')
os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists
total_subjects = tract_specific_data['subject_id'].nunique()

# Group by 'tractID' and count unique 'subject_ID'
tract_counts = tract_specific_data.groupby('tractID')['subject_id'].nunique()

# Filter 'tractID' that are present for every 'subject_ID'
valid_tracts = tract_counts[tract_counts >= 100].index

# Filter the original dataframe
filtered_data = tract_specific_data[tract_specific_data['tractID'].isin(valid_tracts)]

# Show the filtered dataframe
print(filtered_data)
filtered_data.to_csv(os.path.join(results_dir, '2024relevant_tracts.csv'))
relevant_tracts_csv = pd.read_csv(os.path.join(results_dir, '2024relevant_tracts.csv'))

```

          subject_id    tractID    dti_fa PTID_R01_Preop PTID_3T_Postop  \
    0     sub-PDa001   ATR_L_fa  0.275715            NaN            NaN   
    1     sub-PDa001   ATR_R_fa  0.304207            NaN            NaN   
    2     sub-PDa001   CGC_L_fa  0.244211            NaN            NaN   
    3     sub-PDa001   CGC_R_fa  0.230977            NaN            NaN   
    4     sub-PDa001   CGH_L_fa  0.208619            NaN            NaN   
    ...          ...        ...       ...            ...            ...   
    2895  sub-PDa411  SLFT_R_fa  0.417561            NaN            NaN   
    2896  sub-PDa411   SLF_L_fa  0.274110            NaN            NaN   
    2897  sub-PDa411   SLF_R_fa  0.280753            NaN            NaN   
    2898  sub-PDa411    UF_L_fa  0.188834            NaN            NaN   
    2899  sub-PDa411    UF_R_fa  0.323441            NaN            NaN   
    
         PTID_Preop_fMRI_CONN      Consent_General Consent_R01  \
    0                     NaN  2016-01-25 00:00:00         NaT   
    1                     NaN  2016-01-25 00:00:00         NaT   
    2                     NaN  2016-01-25 00:00:00         NaT   
    3                     NaN  2016-01-25 00:00:00         NaT   
    4                     NaN  2016-01-25 00:00:00         NaT   
    ...                   ...                  ...         ...   
    2895                   91  2023-09-01 00:00:00         NaT   
    2896                   91  2023-09-01 00:00:00         NaT   
    2897                   91  2023-09-01 00:00:00         NaT   
    2898                   91  2023-09-01 00:00:00         NaT   
    2899                   91  2023-09-01 00:00:00         NaT   
    
         R01_Prelim_Analysis Preop_R01_Candidate  ...  \
    0                      N                   N  ...   
    1                      N                   N  ...   
    2                      N                   N  ...   
    3                      N                   N  ...   
    4                      N                   N  ...   
    ...                  ...                 ...  ...   
    2895                   N                   Y  ...   
    2896                   N                   Y  ...   
    2897                   N                   Y  ...   
    2898                   N                   Y  ...   
    2899                   N                   Y  ...   
    
         R01_Beck Anxiety Inventory (BAI): Raw   \
    0                                       NaN   
    1                                       NaN   
    2                                       NaN   
    3                                       NaN   
    4                                       NaN   
    ...                                     ...   
    2895                                    NaN   
    2896                                    NaN   
    2897                                    NaN   
    2898                                    NaN   
    2899                                    NaN   
    
         R01_Beck Anxiety Inventory (BAI): T-Score  \
    0                                          NaN   
    1                                          NaN   
    2                                          NaN   
    3                                          NaN   
    4                                          NaN   
    ...                                        ...   
    2895                                       NaN   
    2896                                       NaN   
    2897                                       NaN   
    2898                                       NaN   
    2899                                       NaN   
    
         R01_Parkinson’s Anxiety Scale (PAS): Raw  \
    0                                         NaN   
    1                                         NaN   
    2                                         NaN   
    3                                         NaN   
    4                                         NaN   
    ...                                       ...   
    2895                                      NaN   
    2896                                      NaN   
    2897                                      NaN   
    2898                                      NaN   
    2899                                      NaN   
    
         R01_Parkinson’s Anxiety Scale (PAS): T-Score  \
    0                                             NaN   
    1                                             NaN   
    2                                             NaN   
    3                                             NaN   
    4                                             NaN   
    ...                                           ...   
    2895                                          NaN   
    2896                                          NaN   
    2897                                          NaN   
    2898                                          NaN   
    2899                                          NaN   
    
         R01_Starkstein Apathy Scale (SAS): Raw  \
    0                                       NaN   
    1                                       NaN   
    2                                       NaN   
    3                                       NaN   
    4                                       NaN   
    ...                                     ...   
    2895                                    NaN   
    2896                                    NaN   
    2897                                    NaN   
    2898                                    NaN   
    2899                                    NaN   
    
         R01_Starkstein Apathy Scale (SAS): T-Score  \
    0                                           NaN   
    1                                           NaN   
    2                                           NaN   
    3                                           NaN   
    4                                           NaN   
    ...                                         ...   
    2895                                        NaN   
    2896                                        NaN   
    2897                                        NaN   
    2898                                        NaN   
    2899                                        NaN   
    
         R01_Non-Motor Symptoms Questionnaire (NMS-Quest): Raw R01_QUIP-A  \
    0                                                   NaN           NaN   
    1                                                   NaN           NaN   
    2                                                   NaN           NaN   
    3                                                   NaN           NaN   
    4                                                   NaN           NaN   
    ...                                                 ...           ...   
    2895                                                NaN           NaN   
    2896                                                NaN           NaN   
    2897                                                NaN           NaN   
    2898                                                NaN           NaN   
    2899                                                NaN           NaN   
    
          R01_Columbia Suicide Severity Rating Scale (C-SSRS): Raw  \
    0                                                   NaN          
    1                                                   NaN          
    2                                                   NaN          
    3                                                   NaN          
    4                                                   NaN          
    ...                                                 ...          
    2895                                                NaN          
    2896                                                NaN          
    2897                                                NaN          
    2898                                                NaN          
    2899                                                NaN          
    
          R01_Neuropsychology_Notes  
    0                           NaN  
    1                           NaN  
    2                           NaN  
    3                           NaN  
    4                           NaN  
    ...                         ...  
    2895                        NaN  
    2896                        NaN  
    2897                        NaN  
    2898                        NaN  
    2899                        NaN  
    
    [2900 rows x 308 columns]


    /var/folders/_t/1k38qqf536x8_n0nbth3c0vm0000gn/T/ipykernel_14277/1631502638.py:15: DtypeWarning: Columns (7,34,35,41,168,179,183,184,189,190,191,197,198,199,209,210,211,213,214,215,216,219,224,246,288) have mixed types. Specify dtype option on import or set low_memory=False.
      relevant_tracts_csv = pd.read_csv("/Users/kjs/Library/CloudStorage/Box-Box/Medical_School/AFQ_Analysis/2024relevant_tracts.csv")



```python
unique_tract_ids = relevant_tracts_csv['tractID'].unique()

# Display the unique values
print(unique_tract_ids)
```

    ['ATR_L_fa' 'ATR_R_fa' 'CGC_L_fa' 'CGC_R_fa' 'CGH_L_fa' 'CGH_R_fa'
     'CST_L_fa' 'CST_R_fa' 'FMA_fa' 'FMI_fa' 'IFOF_L_fa' 'IFOF_R_fa'
     'ILF_L_fa' 'ILF_R_fa' 'SLFT_L_fa' 'SLFT_R_fa' 'SLF_L_fa' 'SLF_R_fa'
     'UF_L_fa' 'UF_R_fa']


### Prepare separate STN and GPi dataframes


```python
relevant_tracts_csv_stn = relevant_tracts_csv[relevant_tracts_csv['Target_L_R'] == 'STN']
relevant_tracts_csv_gpi = relevant_tracts_csv[relevant_tracts_csv['Target_L_R'] == 'GPi']

relevant_tracts_csv_gpi
relevant_tracts_csv_stn
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0.1</th>
      <th>Unnamed: 0</th>
      <th>subject_id</th>
      <th>tractID</th>
      <th>dti_fa</th>
      <th>PTID_R01_Preop</th>
      <th>PTID_3T_Postop</th>
      <th>PTID_Preop_fMRI_CONN</th>
      <th>Consent_General</th>
      <th>Consent_R01</th>
      <th>...</th>
      <th>R01_Beck Anxiety Inventory (BAI): Raw</th>
      <th>R01_Beck Anxiety Inventory (BAI): T-Score</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): Raw</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): T-Score</th>
      <th>R01_Starkstein Apathy Scale (SAS): Raw</th>
      <th>R01_Starkstein Apathy Scale (SAS): T-Score</th>
      <th>R01_Non-Motor Symptoms Questionnaire (NMS-Quest): Raw</th>
      <th>R01_QUIP-A</th>
      <th>R01_Columbia Suicide Severity Rating Scale (C-SSRS): Raw</th>
      <th>R01_Neuropsychology_Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>sub-PDa001</td>
      <td>ATR_L_fa</td>
      <td>0.275715</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>sub-PDa001</td>
      <td>ATR_R_fa</td>
      <td>0.304207</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>sub-PDa001</td>
      <td>CGC_L_fa</td>
      <td>0.244211</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>sub-PDa001</td>
      <td>CGC_R_fa</td>
      <td>0.230977</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>sub-PDa001</td>
      <td>CGH_L_fa</td>
      <td>0.208619</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2835</th>
      <td>2835</td>
      <td>2835</td>
      <td>sub-PDa401</td>
      <td>SLFT_R_fa</td>
      <td>0.461140</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78.0</td>
      <td>2023-04-17 00:00:00</td>
      <td>2024-07-08</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2836</th>
      <td>2836</td>
      <td>2836</td>
      <td>sub-PDa401</td>
      <td>SLF_L_fa</td>
      <td>0.348691</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78.0</td>
      <td>2023-04-17 00:00:00</td>
      <td>2024-07-08</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2837</th>
      <td>2837</td>
      <td>2837</td>
      <td>sub-PDa401</td>
      <td>SLF_R_fa</td>
      <td>0.333642</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78.0</td>
      <td>2023-04-17 00:00:00</td>
      <td>2024-07-08</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2838</th>
      <td>2838</td>
      <td>2838</td>
      <td>sub-PDa401</td>
      <td>UF_L_fa</td>
      <td>0.279582</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78.0</td>
      <td>2023-04-17 00:00:00</td>
      <td>2024-07-08</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2839</th>
      <td>2839</td>
      <td>2839</td>
      <td>sub-PDa401</td>
      <td>UF_R_fa</td>
      <td>0.208392</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78.0</td>
      <td>2023-04-17 00:00:00</td>
      <td>2024-07-08</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1631 rows × 310 columns</p>
</div>



### Create patient_df, a dataframe containing all patients with tract data and whole brain data in columns.


```python
pivot_tract_dti = relevant_tracts_csv.pivot(index='subject_id', columns='tractID', values='dti_fa')

# Drop tractID and dti_fa from the original dataframe to avoid duplication
reduced_df = relevant_tracts_csv.drop(columns=['tractID', 'dti_fa'])

# Aggregate the other columns by taking the first non-null value for each subject_id
aggregated_df = reduced_df.groupby('subject_id').first()

# Join the aggregated dataframe with the pivoted tractID-dti_fa dataframe
patient_df = aggregated_df.join(pivot_tract_dti)

# Rename columns to have consistent naming for the merge
metadata_dir = os.path.join('..', 'data', 'metadata')
wholebrain = pd.read_csv(os.path.join(metadata_dir, 'FA_erosion_results.csv'))
wholebrain['subject_id'] = wholebrain['ID'].str.replace('_fa.nii.gz', '')
merged_df = pd.merge(patient_df, wholebrain.drop(columns=['ID']), on='subject_id', how='left')

# Merge the dataframes on 'subject_id'
patient_df = pd.merge(patient_df, wholebrain, on='subject_id', how='left')

columns_to_adjust = [
    'UPDRSIIIpre_Percent_TOTAL_V',
    'UPDRSIIIpre_Percent_TOTAL',
    'UPDRSIIIpre_Percent_BRADY',
    'UPDRSIIIpre_Percent_TREMOR',
    'UPDRSIIIpre_Percent_AXIAL',
    'UPDRSIIIpre_Percent_AXIAL_V',
    'UPDRSIIIpre_Percent_RIGIDITY'
]

# Creating new columns with adjusted values
for column in columns_to_adjust:
    patient_df[column + '_adjusted'] = patient_df[column] + 7

# Display the first few rows of the merged dataframe
results_dir = os.path.join('..', 'results')
os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists
patient_df.to_csv(os.path.join(results_dir, 'patient_df.csv'))

patient_df['subject_id']
```




    0      sub-PDa001
    1      sub-PDa002
    2      sub-PDa003
    3      sub-PDa005
    4      sub-PDa006
              ...    
    141    sub-PDa399
    142    sub-PDa401
    143    sub-PDa402
    144    sub-PDa409
    145    sub-PDa411
    Name: subject_id, Length: 146, dtype: object



# Add UPDRS subjects without MDS scores


```python
# List of subject_ids to replace
target_subjects = ['sub-PDa002', 'sub-PDa003', 'sub-PDa009', 'sub-PDa010', 'sub-PDa016', 
                   'sub-PDa018', 'sub-PDa019', 'sub-PDa035', 'sub-PDa063', 'sub-PDa070', 
                   'sub-PDa125', 'sub-PDa347']

# Columns with their corresponding adjusted versions in patient_df
columns_to_adjust = [
    ('MDSUPDRSIIIpre_Percent_TOTAL_V', 'UPDRSIIIpre_Percent_TOTAL_V_adjusted'),
    ('MDSUPDRSIIIpre_Percent_TOTAL', 'UPDRSIIIpre_Percent_TOTAL_adjusted'),
    ('MDSUPDRSIIIpre_Percent_BRADY', 'UPDRSIIIpre_Percent_BRADY_adjusted'),
    ('MDSUPDRSIIIpre_Percent_TREMOR', 'UPDRSIIIpre_Percent_TREMOR_adjusted'),
    ('MDSUPDRSIIIpre_Percent_AXIAL', 'UPDRSIIIpre_Percent_AXIAL_adjusted'),
    ('MDSUPDRSIIIpre_Percent_AXIAL_V', 'UPDRSIIIpre_Percent_AXIAL_V_adjusted'),
    ('MDSUPDRSIIIpre_Percent_RIGIDITY', 'UPDRSIIIpre_Percent_RIGIDITY_adjusted')
]

# Capture substituted values for output
substituted_values = []

# Replace MDSUPDRS values in patient_df with adjusted UPDRS values and track changes
for mdsupdrs_col, updrs_adjusted_col in columns_to_adjust:
    for subject in target_subjects:
        if subject in patient_df['subject_id'].values:
            # Fetch the adjusted value
            adjusted_value = patient_df.loc[patient_df['subject_id'] == subject, updrs_adjusted_col].values[0]
            original_value = patient_df.loc[patient_df['subject_id'] == subject, mdsupdrs_col].values[0]
            substituted_values.append((subject, mdsupdrs_col, original_value, adjusted_value))
            # Replace the original MDSUPDRS value with the adjusted UPDRS value
            patient_df.loc[patient_df['subject_id'] == subject, mdsupdrs_col] = adjusted_value

# Output the substituted values
for substitution in substituted_values:
    print(f"Subject ID: {substitution[0]}, Column: {substitution[1]}, Original MDSUPDRS Value: {substitution[2]}, Adjusted UPDRS Value: {substitution[3]}")


```

    Subject ID: sub-PDa002, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 84.14285714285715, Adjusted UPDRS Value: 84.14285714285715
    Subject ID: sub-PDa003, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 84.08333333333334, Adjusted UPDRS Value: 84.08333333333334
    Subject ID: sub-PDa009, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 85.26086956521739, Adjusted UPDRS Value: 85.26086956521739
    Subject ID: sub-PDa010, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 70.63636363636363, Adjusted UPDRS Value: 70.63636363636363
    Subject ID: sub-PDa016, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 68.11111111111111, Adjusted UPDRS Value: 68.11111111111111
    Subject ID: sub-PDa018, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 34.586206896551715, Adjusted UPDRS Value: 34.586206896551715
    Subject ID: sub-PDa019, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 75.0, Adjusted UPDRS Value: 75.0
    Subject ID: sub-PDa035, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 58.35135135135135, Adjusted UPDRS Value: 58.35135135135135
    Subject ID: sub-PDa063, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 44.03703704, Adjusted UPDRS Value: 44.03703704
    Subject ID: sub-PDa070, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 43.84210526, Adjusted UPDRS Value: 43.84210526
    Subject ID: sub-PDa125, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 45.88888889, Adjusted UPDRS Value: 45.88888889
    Subject ID: sub-PDa347, Column: MDSUPDRSIIIpre_Percent_TOTAL_V, Original MDSUPDRS Value: 63.66666667, Adjusted UPDRS Value: 63.66666667
    Subject ID: sub-PDa002, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 81.35897435897436, Adjusted UPDRS Value: 81.35897435897436
    Subject ID: sub-PDa003, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 84.27272727272727, Adjusted UPDRS Value: 84.27272727272727
    Subject ID: sub-PDa009, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 79.72727272727273, Adjusted UPDRS Value: 79.72727272727273
    Subject ID: sub-PDa010, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 73.10169491525424, Adjusted UPDRS Value: 73.10169491525424
    Subject ID: sub-PDa016, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 57.0, Adjusted UPDRS Value: 57.0
    Subject ID: sub-PDa018, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 47.54054054054054, Adjusted UPDRS Value: 47.54054054054054
    Subject ID: sub-PDa019, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 83.47058824, Adjusted UPDRS Value: 83.47058824
    Subject ID: sub-PDa035, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 52.83333333333333, Adjusted UPDRS Value: 52.83333333333333
    Subject ID: sub-PDa063, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 42.29411765, Adjusted UPDRS Value: 42.29411765
    Subject ID: sub-PDa070, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 51.0, Adjusted UPDRS Value: 51.0
    Subject ID: sub-PDa125, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 59.17391304, Adjusted UPDRS Value: 59.17391304
    Subject ID: sub-PDa347, Column: MDSUPDRSIIIpre_Percent_TOTAL, Original MDSUPDRS Value: 56.29577465, Adjusted UPDRS Value: 56.29577465
    Subject ID: sub-PDa002, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 78.42857142857143, Adjusted UPDRS Value: 78.42857142857143
    Subject ID: sub-PDa003, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 77.83333333333334, Adjusted UPDRS Value: 77.83333333333334
    Subject ID: sub-PDa009, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 87.0, Adjusted UPDRS Value: 87.0
    Subject ID: sub-PDa010, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 65.33333333333334, Adjusted UPDRS Value: 65.33333333333334
    Subject ID: sub-PDa016, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 84.77777777777779, Adjusted UPDRS Value: 84.77777777777779
    Subject ID: sub-PDa018, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 33.92307692307692, Adjusted UPDRS Value: 33.92307692307692
    Subject ID: sub-PDa019, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 77.58823529, Adjusted UPDRS Value: 77.58823529
    Subject ID: sub-PDa035, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 57.0, Adjusted UPDRS Value: 57.0
    Subject ID: sub-PDa063, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 47.0, Adjusted UPDRS Value: 47.0
    Subject ID: sub-PDa070, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 73.66666667, Adjusted UPDRS Value: 73.66666667
    Subject ID: sub-PDa125, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 47.0, Adjusted UPDRS Value: 47.0
    Subject ID: sub-PDa347, Column: MDSUPDRSIIIpre_Percent_BRADY, Original MDSUPDRS Value: 48.37931034, Adjusted UPDRS Value: 48.37931034
    Subject ID: sub-PDa002, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: 84.77777777777779, Adjusted UPDRS Value: 84.77777777777779
    Subject ID: sub-PDa003, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: 90.33333333333334, Adjusted UPDRS Value: 90.33333333333334
    Subject ID: sub-PDa009, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: nan, Adjusted UPDRS Value: nan
    Subject ID: sub-PDa010, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: 92.71428571428572, Adjusted UPDRS Value: 92.71428571428571
    Subject ID: sub-PDa016, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: 40.33333333333333, Adjusted UPDRS Value: 40.33333333333333
    Subject ID: sub-PDa018, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: 7.0, Adjusted UPDRS Value: 7.0
    Subject ID: sub-PDa019, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: 107.0, Adjusted UPDRS Value: 107.0
    Subject ID: sub-PDa035, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: 59.94117647058824, Adjusted UPDRS Value: 59.94117647058824
    Subject ID: sub-PDa063, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: 47.0, Adjusted UPDRS Value: 47.0
    Subject ID: sub-PDa070, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: 23.66666667, Adjusted UPDRS Value: 23.66666667
    Subject ID: sub-PDa125, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: nan, Adjusted UPDRS Value: nan
    Subject ID: sub-PDa347, Column: MDSUPDRSIIIpre_Percent_TREMOR, Original MDSUPDRS Value: 83.0, Adjusted UPDRS Value: 83.0
    Subject ID: sub-PDa002, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 107.0, Adjusted UPDRS Value: 107.0
    Subject ID: sub-PDa003, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 84.77777777777779, Adjusted UPDRS Value: 84.77777777777779
    Subject ID: sub-PDa009, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 82.0, Adjusted UPDRS Value: 82.0
    Subject ID: sub-PDa010, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 62.55555555555556, Adjusted UPDRS Value: 62.55555555555556
    Subject ID: sub-PDa016, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 73.66666666666666, Adjusted UPDRS Value: 73.66666666666666
    Subject ID: sub-PDa018, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 87.0, Adjusted UPDRS Value: 87.0
    Subject ID: sub-PDa019, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 57.0, Adjusted UPDRS Value: 57.0
    Subject ID: sub-PDa035, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 57.0, Adjusted UPDRS Value: 57.0
    Subject ID: sub-PDa063, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 7.0, Adjusted UPDRS Value: 7.0
    Subject ID: sub-PDa070, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 107.0, Adjusted UPDRS Value: 107.0
    Subject ID: sub-PDa125, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 73.66666667, Adjusted UPDRS Value: 73.66666667
    Subject ID: sub-PDa347, Column: MDSUPDRSIIIpre_Percent_AXIAL, Original MDSUPDRS Value: 64.14285713999999, Adjusted UPDRS Value: 64.14285713999999
    Subject ID: sub-PDa002, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 107.0, Adjusted UPDRS Value: 107.0
    Subject ID: sub-PDa003, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 82.0, Adjusted UPDRS Value: 82.0
    Subject ID: sub-PDa009, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 84.77777777777779, Adjusted UPDRS Value: 84.77777777777779
    Subject ID: sub-PDa010, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 49.857142857142854, Adjusted UPDRS Value: 49.857142857142854
    Subject ID: sub-PDa016, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 73.66666666666666, Adjusted UPDRS Value: 73.66666666666666
    Subject ID: sub-PDa018, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 82.0, Adjusted UPDRS Value: 82.0
    Subject ID: sub-PDa019, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 47.0, Adjusted UPDRS Value: 47.0
    Subject ID: sub-PDa035, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 57.0, Adjusted UPDRS Value: 57.0
    Subject ID: sub-PDa063, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 7.0, Adjusted UPDRS Value: 7.0
    Subject ID: sub-PDa070, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 107.0, Adjusted UPDRS Value: 107.0
    Subject ID: sub-PDa125, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 67.0, Adjusted UPDRS Value: 67.0
    Subject ID: sub-PDa347, Column: MDSUPDRSIIIpre_Percent_AXIAL_V, Original MDSUPDRS Value: 64.14285713999999, Adjusted UPDRS Value: 64.14285713999999
    Subject ID: sub-PDa002, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 40.33333333333333, Adjusted UPDRS Value: 40.33333333333333
    Subject ID: sub-PDa003, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 83.47058823529412, Adjusted UPDRS Value: 83.47058823529412
    Subject ID: sub-PDa009, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 64.14285714285714, Adjusted UPDRS Value: 64.14285714285714
    Subject ID: sub-PDa010, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 76.23076923076923, Adjusted UPDRS Value: 76.23076923076923
    Subject ID: sub-PDa016, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 7.0, Adjusted UPDRS Value: 7.0
    Subject ID: sub-PDa018, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 92.71428571428572, Adjusted UPDRS Value: 92.71428571428571
    Subject ID: sub-PDa019, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 107.0, Adjusted UPDRS Value: 107.0
    Subject ID: sub-PDa035, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 34.272727272727266, Adjusted UPDRS Value: 34.272727272727266
    Subject ID: sub-PDa063, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 35.571428569999995, Adjusted UPDRS Value: 35.571428569999995
    Subject ID: sub-PDa070, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 73.66666667, Adjusted UPDRS Value: 73.66666667
    Subject ID: sub-PDa125, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 107.0, Adjusted UPDRS Value: 107.0
    Subject ID: sub-PDa347, Column: MDSUPDRSIIIpre_Percent_RIGIDITY, Original MDSUPDRS Value: 16.090909091, Adjusted UPDRS Value: 16.090909091



```python
# Pivot relevant_tracts_csv for 'tractID' as columns and 'dti_fa' as values with 'subject_id' as index
pivot_tract_dti = relevant_tracts_csv.pivot(index='subject_id', columns='tractID', values='dti_fa')

# Drop 'tractID' and 'dti_fa' from the original dataframe to avoid duplication
reduced_df = relevant_tracts_csv.drop(columns=['tractID', 'dti_fa'])

# Aggregate other columns by taking the first non-null value for each 'subject_id'
aggregated_df = reduced_df.groupby('subject_id').first()

# Join the aggregated dataframe with the pivoted tractID-dti_fa dataframe
patient_df = aggregated_df.join(pivot_tract_dti)

# Load wholebrain data and clean the 'subject_id' for merging
metadata_dir = os.path.join('..', 'data', 'metadata')
wholebrain_path = os.path.join(metadata_dir, 'FA_erosion_results.csv')
# In this environment, we will skip loading `wholebrain`, assuming it would be merged as in the instructions

# Columns requiring adjustments
columns_to_adjust = [
    'UPDRSIIIpre_Percent_TOTAL_V',
    'UPDRSIIIpre_Percent_TOTAL',
    'UPDRSIIIpre_Percent_BRADY',
    'UPDRSIIIpre_Percent_TREMOR',
    'UPDRSIIIpre_Percent_AXIAL',
    'UPDRSIIIpre_Percent_AXIAL_V',
    'UPDRSIIIpre_Percent_RIGIDITY'
]

# Creating new columns with adjusted values
for column in columns_to_adjust:
    patient_df[column + '_adjusted'] = patient_df[column] + 7

# Define target subjects and columns for MDSUPDRS substitution process
target_subjects = [
    'sub-PDa002', 'sub-PDa003', 'sub-PDa009', 'sub-PDa010', 'sub-PDa016',
    'sub-PDa018', 'sub-PDa019', 'sub-PDa035', 'sub-PDa063', 'sub-PDa070',
    'sub-PDa125', 'sub-PDa347'
]
mdsupdrs_columns_to_adjust = [
    ('MDSUPDRSIIIpre_Percent_TOTAL_V', 'UPDRSIIIpre_Percent_TOTAL_V_adjusted'),
    ('MDSUPDRSIIIpre_Percent_TOTAL', 'UPDRSIIIpre_Percent_TOTAL_adjusted'),
    ('MDSUPDRSIIIpre_Percent_BRADY', 'UPDRSIIIpre_Percent_BRADY_adjusted'),
    ('MDSUPDRSIIIpre_Percent_TREMOR', 'UPDRSIIIpre_Percent_TREMOR_adjusted'),
    ('MDSUPDRSIIIpre_Percent_AXIAL', 'UPDRSIIIpre_Percent_AXIAL_adjusted'),
    ('MDSUPDRSIIIpre_Percent_AXIAL_V', 'UPDRSIIIpre_Percent_AXIAL_V_adjusted'),
    ('MDSUPDRSIIIpre_Percent_RIGIDITY', 'UPDRSIIIpre_Percent_RIGIDITY_adjusted')
]

# Track substituted values for reporting
substituted_values = []

# Replace MDSUPDRS values in patient_df with adjusted UPDRS values
for mdsupdrs_col, updrs_adjusted_col in mdsupdrs_columns_to_adjust:
    for subject in target_subjects:
        if subject in patient_df.index:
            # Fetch the adjusted value
            adjusted_value = patient_df.loc[subject, updrs_adjusted_col]
            original_value = patient_df.loc[subject, mdsupdrs_col]
            substituted_values.append((subject, mdsupdrs_col, original_value, adjusted_value))
            # Replace original MDSUPDRS value with adjusted UPDRS value
            patient_df.loc[subject, mdsupdrs_col] = adjusted_value

# Convert substituted values into a DataFrame for clear presentation
substituted_values_df = pd.DataFrame(
    substituted_values, columns=['Subject ID', 'Column', 'Original MDSUPDRS Value', 'Adjusted UPDRS Value']
)

for index, row in relevant_tracts_csv.iterrows():
    subject = row['subject_id']
    
    # Only update for target subjects
    if subject in target_subjects:
        # Replace each MDSUPDRS column with the corresponding adjusted value
        for mdsupdrs_col, updrs_adjusted_col in mdsupdrs_columns_to_adjust:
            # Check if adjusted value exists in patient_df
            if updrs_adjusted_col in patient_df.columns:
                adjusted_value = patient_df.loc[subject, updrs_adjusted_col]
                # Update relevant_tracts_csv with the adjusted value
                relevant_tracts_csv.at[index, mdsupdrs_col] = adjusted_value


results_dir = os.path.join('..', 'results')
os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists
relevant_tracts_csv.to_csv(os.path.join(results_dir, '2024relevant_tracts.csv'))
relevant_tracts_csv = pd.read_csv(os.path.join(results_dir, '2024relevant_tracts.csv'))
relevant_tracts_csv
```

    /var/folders/_t/1k38qqf536x8_n0nbth3c0vm0000gn/T/ipykernel_14277/3905446053.py:82: DtypeWarning: Columns (7,34,35,41,168,179,183,184,189,190,191,197,198,199,209,210,211,213,214,215,216,219,224,246,288) have mixed types. Specify dtype option on import or set low_memory=False.
      relevant_tracts_csv = pd.read_csv("/Users/kjs/Library/CloudStorage/Box-Box/Medical_School/AFQ_Analysis/2024relevant_tracts.csv")





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0.1</th>
      <th>Unnamed: 0</th>
      <th>subject_id</th>
      <th>tractID</th>
      <th>dti_fa</th>
      <th>PTID_R01_Preop</th>
      <th>PTID_3T_Postop</th>
      <th>PTID_Preop_fMRI_CONN</th>
      <th>Consent_General</th>
      <th>Consent_R01</th>
      <th>...</th>
      <th>R01_Beck Anxiety Inventory (BAI): Raw</th>
      <th>R01_Beck Anxiety Inventory (BAI): T-Score</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): Raw</th>
      <th>R01_Parkinson’s Anxiety Scale (PAS): T-Score</th>
      <th>R01_Starkstein Apathy Scale (SAS): Raw</th>
      <th>R01_Starkstein Apathy Scale (SAS): T-Score</th>
      <th>R01_Non-Motor Symptoms Questionnaire (NMS-Quest): Raw</th>
      <th>R01_QUIP-A</th>
      <th>R01_Columbia Suicide Severity Rating Scale (C-SSRS): Raw</th>
      <th>R01_Neuropsychology_Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>sub-PDa001</td>
      <td>ATR_L_fa</td>
      <td>0.275715</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>sub-PDa001</td>
      <td>ATR_R_fa</td>
      <td>0.304207</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>sub-PDa001</td>
      <td>CGC_L_fa</td>
      <td>0.244211</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>sub-PDa001</td>
      <td>CGC_R_fa</td>
      <td>0.230977</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>sub-PDa001</td>
      <td>CGH_L_fa</td>
      <td>0.208619</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-25 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2895</th>
      <td>2895</td>
      <td>2895</td>
      <td>sub-PDa411</td>
      <td>SLFT_R_fa</td>
      <td>0.417561</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.0</td>
      <td>2023-09-01 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2896</th>
      <td>2896</td>
      <td>2896</td>
      <td>sub-PDa411</td>
      <td>SLF_L_fa</td>
      <td>0.274110</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.0</td>
      <td>2023-09-01 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2897</th>
      <td>2897</td>
      <td>2897</td>
      <td>sub-PDa411</td>
      <td>SLF_R_fa</td>
      <td>0.280753</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.0</td>
      <td>2023-09-01 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2898</th>
      <td>2898</td>
      <td>2898</td>
      <td>sub-PDa411</td>
      <td>UF_L_fa</td>
      <td>0.188834</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.0</td>
      <td>2023-09-01 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2899</th>
      <td>2899</td>
      <td>2899</td>
      <td>sub-PDa411</td>
      <td>UF_R_fa</td>
      <td>0.323441</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.0</td>
      <td>2023-09-01 00:00:00</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2900 rows × 310 columns</p>
</div>



# Remove patients with missing data from either delta_ledd1 or delta_ledd2
patient_df = patient_df.dropna(subset=['delta_ledd1', 'delta_ledd2'])
patient_df['FA_0']

```python
num_values = patient_df["MDSUPDRSIIIpre_Percent_TOTAL_V"].notna().sum()

print(f"Number of values in 'MDSUPDRSIIIpre_Percent_TOTAL': {num_values}")
```

    Number of values in 'MDSUPDRSIIIpre_Percent_TOTAL': 133



```python
metadata_dir = os.path.join('..', 'data', 'metadata')
wholebrain = pd.read_csv(os.path.join(metadata_dir, 'FA_erosion_results.csv'))
wholebrain['subject_id'] = wholebrain['ID'].str.replace('_fa.nii.gz', '')
merged_df = pd.merge(patient_df, wholebrain.drop(columns=['ID']), on='subject_id', how='left')

# Merge the dataframes on 'subject_id'
patient_df = pd.merge(patient_df, wholebrain, on='subject_id', how='left')
```


```python
patient_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>Unnamed: 0.1</th>
      <th>Unnamed: 0</th>
      <th>PTID_R01_Preop</th>
      <th>PTID_3T_Postop</th>
      <th>PTID_Preop_fMRI_CONN</th>
      <th>Consent_General</th>
      <th>Consent_R01</th>
      <th>R01_Prelim_Analysis</th>
      <th>Preop_R01_Candidate</th>
      <th>...</th>
      <th>UPDRSIIIpre_Percent_AXIAL_adjusted</th>
      <th>UPDRSIIIpre_Percent_AXIAL_V_adjusted</th>
      <th>UPDRSIIIpre_Percent_RIGIDITY_adjusted</th>
      <th>ID</th>
      <th>FA_0</th>
      <th>FA_5</th>
      <th>FA_10</th>
      <th>FA_15</th>
      <th>FA_20</th>
      <th>FA_25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-PDa001</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-01-25 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa001</td>
      <td>0.353384</td>
      <td>0.352412</td>
      <td>0.352839</td>
      <td>0.357490</td>
      <td>0.358653</td>
      <td>0.361728</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-PDa002</td>
      <td>20</td>
      <td>20</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-02-01 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>107.000000</td>
      <td>107.0</td>
      <td>40.333333</td>
      <td>sub-PDa002</td>
      <td>0.363276</td>
      <td>0.362649</td>
      <td>0.363395</td>
      <td>0.369296</td>
      <td>0.370871</td>
      <td>0.374030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-PDa003</td>
      <td>40</td>
      <td>40</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-02-29 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>84.777778</td>
      <td>82.0</td>
      <td>83.470588</td>
      <td>sub-PDa003</td>
      <td>0.372342</td>
      <td>0.372967</td>
      <td>0.373939</td>
      <td>0.379770</td>
      <td>0.381452</td>
      <td>0.385542</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-PDa005</td>
      <td>56</td>
      <td>56</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-03-15 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa005</td>
      <td>0.357305</td>
      <td>0.360006</td>
      <td>0.363166</td>
      <td>0.369126</td>
      <td>0.370489</td>
      <td>0.374036</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-PDa006</td>
      <td>76</td>
      <td>76</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2022-11-01 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa006</td>
      <td>0.388189</td>
      <td>0.385523</td>
      <td>0.386879</td>
      <td>0.392743</td>
      <td>0.394241</td>
      <td>0.398246</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>141</th>
      <td>sub-PDa399</td>
      <td>2800</td>
      <td>2800</td>
      <td>None</td>
      <td>None</td>
      <td>87.0</td>
      <td>2023-07-24 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa399</td>
      <td>0.392420</td>
      <td>0.396018</td>
      <td>0.397934</td>
      <td>0.403989</td>
      <td>0.405517</td>
      <td>0.409068</td>
    </tr>
    <tr>
      <th>142</th>
      <td>sub-PDa401</td>
      <td>2820</td>
      <td>2820</td>
      <td>None</td>
      <td>None</td>
      <td>78.0</td>
      <td>2023-04-17 00:00:00</td>
      <td>2024-07-08</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa401</td>
      <td>0.385118</td>
      <td>0.385183</td>
      <td>0.385560</td>
      <td>0.388874</td>
      <td>0.389868</td>
      <td>0.392674</td>
    </tr>
    <tr>
      <th>143</th>
      <td>sub-PDa402</td>
      <td>2840</td>
      <td>2840</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2023-05-12 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa402</td>
      <td>0.357774</td>
      <td>0.357437</td>
      <td>0.357590</td>
      <td>0.358806</td>
      <td>0.359423</td>
      <td>0.361289</td>
    </tr>
    <tr>
      <th>144</th>
      <td>sub-PDa409</td>
      <td>2860</td>
      <td>2860</td>
      <td>None</td>
      <td>None</td>
      <td>90.0</td>
      <td>2023-07-24 00:00:00</td>
      <td>2024-04-23</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa409</td>
      <td>0.391404</td>
      <td>0.392053</td>
      <td>0.392887</td>
      <td>0.397328</td>
      <td>0.398567</td>
      <td>0.401733</td>
    </tr>
    <tr>
      <th>145</th>
      <td>sub-PDa411</td>
      <td>2880</td>
      <td>2880</td>
      <td>None</td>
      <td>None</td>
      <td>91.0</td>
      <td>2023-09-01 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa411</td>
      <td>0.394639</td>
      <td>0.395941</td>
      <td>0.396622</td>
      <td>0.399024</td>
      <td>0.399697</td>
      <td>0.401606</td>
    </tr>
  </tbody>
</table>
<p>146 rows × 342 columns</p>
</div>



### Create STN-specific dataframe


```python
results_dir = os.path.join('..', 'results')
os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists
patient_df_stn = patient_df[patient_df['Target_L_R'].str.lower().isin(['stn'])]
patient_df_stn.to_csv(os.path.join(results_dir, '2024fulldf_stn.csv'))
patient_df_stn
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>Unnamed: 0.1</th>
      <th>Unnamed: 0</th>
      <th>PTID_R01_Preop</th>
      <th>PTID_3T_Postop</th>
      <th>PTID_Preop_fMRI_CONN</th>
      <th>Consent_General</th>
      <th>Consent_R01</th>
      <th>R01_Prelim_Analysis</th>
      <th>Preop_R01_Candidate</th>
      <th>...</th>
      <th>UPDRSIIIpre_Percent_AXIAL_adjusted</th>
      <th>UPDRSIIIpre_Percent_AXIAL_V_adjusted</th>
      <th>UPDRSIIIpre_Percent_RIGIDITY_adjusted</th>
      <th>ID</th>
      <th>FA_0</th>
      <th>FA_5</th>
      <th>FA_10</th>
      <th>FA_15</th>
      <th>FA_20</th>
      <th>FA_25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-PDa001</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-01-25 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa001</td>
      <td>0.353384</td>
      <td>0.352412</td>
      <td>0.352839</td>
      <td>0.357490</td>
      <td>0.358653</td>
      <td>0.361728</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-PDa002</td>
      <td>20</td>
      <td>20</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-02-01 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>107.000000</td>
      <td>107.0</td>
      <td>40.333333</td>
      <td>sub-PDa002</td>
      <td>0.363276</td>
      <td>0.362649</td>
      <td>0.363395</td>
      <td>0.369296</td>
      <td>0.370871</td>
      <td>0.374030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-PDa003</td>
      <td>40</td>
      <td>40</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-02-29 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>84.777778</td>
      <td>82.0</td>
      <td>83.470588</td>
      <td>sub-PDa003</td>
      <td>0.372342</td>
      <td>0.372967</td>
      <td>0.373939</td>
      <td>0.379770</td>
      <td>0.381452</td>
      <td>0.385542</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-PDa005</td>
      <td>56</td>
      <td>56</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-03-15 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa005</td>
      <td>0.357305</td>
      <td>0.360006</td>
      <td>0.363166</td>
      <td>0.369126</td>
      <td>0.370489</td>
      <td>0.374036</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub-PDa007</td>
      <td>96</td>
      <td>96</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-03-28 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa007</td>
      <td>0.360593</td>
      <td>0.359513</td>
      <td>0.359605</td>
      <td>0.363062</td>
      <td>0.364148</td>
      <td>0.367386</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>132</th>
      <td>sub-PDa371</td>
      <td>2620</td>
      <td>2620</td>
      <td>None</td>
      <td>PDpf014</td>
      <td>70.0</td>
      <td>2022-09-19 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa371</td>
      <td>0.366148</td>
      <td>0.366847</td>
      <td>0.367884</td>
      <td>0.371978</td>
      <td>0.373119</td>
      <td>0.376099</td>
    </tr>
    <tr>
      <th>133</th>
      <td>sub-PDa372</td>
      <td>2640</td>
      <td>2640</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2022-10-11 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa372</td>
      <td>0.380893</td>
      <td>0.381521</td>
      <td>0.382342</td>
      <td>0.385526</td>
      <td>0.386490</td>
      <td>0.389100</td>
    </tr>
    <tr>
      <th>137</th>
      <td>sub-PDa376</td>
      <td>2720</td>
      <td>2720</td>
      <td>PDpm002</td>
      <td>None</td>
      <td>73.0</td>
      <td>2022-11-21 00:00:00</td>
      <td>2023-08-30</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa376</td>
      <td>0.376811</td>
      <td>0.377509</td>
      <td>0.378496</td>
      <td>0.381673</td>
      <td>0.382572</td>
      <td>0.384980</td>
    </tr>
    <tr>
      <th>141</th>
      <td>sub-PDa399</td>
      <td>2800</td>
      <td>2800</td>
      <td>None</td>
      <td>None</td>
      <td>87.0</td>
      <td>2023-07-24 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa399</td>
      <td>0.392420</td>
      <td>0.396018</td>
      <td>0.397934</td>
      <td>0.403989</td>
      <td>0.405517</td>
      <td>0.409068</td>
    </tr>
    <tr>
      <th>142</th>
      <td>sub-PDa401</td>
      <td>2820</td>
      <td>2820</td>
      <td>None</td>
      <td>None</td>
      <td>78.0</td>
      <td>2023-04-17 00:00:00</td>
      <td>2024-07-08</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa401</td>
      <td>0.385118</td>
      <td>0.385183</td>
      <td>0.385560</td>
      <td>0.388874</td>
      <td>0.389868</td>
      <td>0.392674</td>
    </tr>
  </tbody>
</table>
<p>82 rows × 342 columns</p>
</div>



### Create GPi-specific dataframe


```python
results_dir = os.path.join('..', 'results')
os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists
patient_df_gpi = patient_df[patient_df['Target_L_R'].str.lower().isin(['gpi'])]
patient_df_gpi.to_csv(os.path.join(results_dir, '2024fulldf_gpi.csv'))
patient_df_gpi
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>Unnamed: 0.1</th>
      <th>Unnamed: 0</th>
      <th>PTID_R01_Preop</th>
      <th>PTID_3T_Postop</th>
      <th>PTID_Preop_fMRI_CONN</th>
      <th>Consent_General</th>
      <th>Consent_R01</th>
      <th>R01_Prelim_Analysis</th>
      <th>Preop_R01_Candidate</th>
      <th>...</th>
      <th>UPDRSIIIpre_Percent_AXIAL_adjusted</th>
      <th>UPDRSIIIpre_Percent_AXIAL_V_adjusted</th>
      <th>UPDRSIIIpre_Percent_RIGIDITY_adjusted</th>
      <th>ID</th>
      <th>FA_0</th>
      <th>FA_5</th>
      <th>FA_10</th>
      <th>FA_15</th>
      <th>FA_20</th>
      <th>FA_25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>sub-PDa006</td>
      <td>76</td>
      <td>76</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2022-11-01 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa006</td>
      <td>0.388189</td>
      <td>0.385523</td>
      <td>0.386879</td>
      <td>0.392743</td>
      <td>0.394241</td>
      <td>0.398246</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sub-PDa011</td>
      <td>176</td>
      <td>176</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-06-14 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa011</td>
      <td>0.381011</td>
      <td>0.380147</td>
      <td>0.381094</td>
      <td>0.386961</td>
      <td>0.388494</td>
      <td>0.392257</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sub-PDa013</td>
      <td>213</td>
      <td>213</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-07-11 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa013</td>
      <td>0.371172</td>
      <td>0.369938</td>
      <td>0.370680</td>
      <td>0.375199</td>
      <td>0.376660</td>
      <td>0.380465</td>
    </tr>
    <tr>
      <th>15</th>
      <td>sub-PDa018</td>
      <td>293</td>
      <td>293</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-05-04 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>87.000000</td>
      <td>82.0</td>
      <td>92.714286</td>
      <td>sub-PDa018</td>
      <td>0.354026</td>
      <td>0.354712</td>
      <td>0.356138</td>
      <td>0.362210</td>
      <td>0.363766</td>
      <td>0.367914</td>
    </tr>
    <tr>
      <th>17</th>
      <td>sub-PDa020</td>
      <td>333</td>
      <td>333</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-10-31 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa020</td>
      <td>0.351310</td>
      <td>0.352045</td>
      <td>0.353396</td>
      <td>0.359254</td>
      <td>0.360639</td>
      <td>0.364315</td>
    </tr>
    <tr>
      <th>19</th>
      <td>sub-PDa022</td>
      <td>373</td>
      <td>373</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-11-21 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa022</td>
      <td>0.355439</td>
      <td>0.355538</td>
      <td>0.356189</td>
      <td>0.361058</td>
      <td>0.362355</td>
      <td>0.365557</td>
    </tr>
    <tr>
      <th>28</th>
      <td>sub-PDa031</td>
      <td>545</td>
      <td>545</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2017-03-24 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa031</td>
      <td>0.365488</td>
      <td>0.365641</td>
      <td>0.366683</td>
      <td>0.372060</td>
      <td>0.373309</td>
      <td>0.376605</td>
    </tr>
    <tr>
      <th>31</th>
      <td>sub-PDa035</td>
      <td>605</td>
      <td>605</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2014-04-14 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>57.000000</td>
      <td>57.0</td>
      <td>34.272727</td>
      <td>sub-PDa035</td>
      <td>0.369642</td>
      <td>0.369519</td>
      <td>0.370427</td>
      <td>0.376783</td>
      <td>0.378447</td>
      <td>0.382253</td>
    </tr>
    <tr>
      <th>35</th>
      <td>sub-PDa039</td>
      <td>680</td>
      <td>680</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2014-06-18 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa039</td>
      <td>0.358539</td>
      <td>0.358441</td>
      <td>0.359489</td>
      <td>0.364449</td>
      <td>0.365759</td>
      <td>0.369014</td>
    </tr>
    <tr>
      <th>46</th>
      <td>sub-PDa055</td>
      <td>900</td>
      <td>900</td>
      <td>None</td>
      <td>None</td>
      <td>6</td>
      <td>2018-06-18 00:00:00</td>
      <td>None</td>
      <td>Y</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa055</td>
      <td>0.374315</td>
      <td>0.374989</td>
      <td>0.375538</td>
      <td>0.377836</td>
      <td>0.378595</td>
      <td>0.380742</td>
    </tr>
    <tr>
      <th>47</th>
      <td>sub-PDa056</td>
      <td>920</td>
      <td>920</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2018-07-02 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa056</td>
      <td>0.392052</td>
      <td>0.393068</td>
      <td>0.394322</td>
      <td>0.399170</td>
      <td>0.400462</td>
      <td>0.403842</td>
    </tr>
    <tr>
      <th>48</th>
      <td>sub-PDa057</td>
      <td>940</td>
      <td>940</td>
      <td>None</td>
      <td>None</td>
      <td>55</td>
      <td>2017-12-11 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa057</td>
      <td>0.380382</td>
      <td>0.380375</td>
      <td>0.380408</td>
      <td>0.382877</td>
      <td>0.383643</td>
      <td>0.385739</td>
    </tr>
    <tr>
      <th>49</th>
      <td>sub-PDa058</td>
      <td>960</td>
      <td>960</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2018-08-27 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa058</td>
      <td>0.368175</td>
      <td>0.367345</td>
      <td>0.367614</td>
      <td>0.372362</td>
      <td>0.373627</td>
      <td>0.376932</td>
    </tr>
    <tr>
      <th>50</th>
      <td>sub-PDa060</td>
      <td>980</td>
      <td>980</td>
      <td>None</td>
      <td>None</td>
      <td>NA \n(fMRI only has one file)</td>
      <td>2018-10-01 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa060</td>
      <td>0.370236</td>
      <td>0.371009</td>
      <td>0.372025</td>
      <td>0.376146</td>
      <td>0.377321</td>
      <td>0.380207</td>
    </tr>
    <tr>
      <th>52</th>
      <td>sub-PDa062</td>
      <td>1020</td>
      <td>1020</td>
      <td>None</td>
      <td>None</td>
      <td>57</td>
      <td>2020-03-04 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa062</td>
      <td>0.380728</td>
      <td>0.381391</td>
      <td>0.382348</td>
      <td>0.386422</td>
      <td>0.387610</td>
      <td>0.390720</td>
    </tr>
    <tr>
      <th>54</th>
      <td>sub-PDa064</td>
      <td>1060</td>
      <td>1060</td>
      <td>None</td>
      <td>None</td>
      <td>58</td>
      <td>2018-11-05 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa064</td>
      <td>0.372926</td>
      <td>0.373664</td>
      <td>0.374416</td>
      <td>0.377857</td>
      <td>0.378839</td>
      <td>0.381522</td>
    </tr>
    <tr>
      <th>56</th>
      <td>sub-PDa066</td>
      <td>1100</td>
      <td>1100</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2018-12-10 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa066</td>
      <td>0.387063</td>
      <td>0.387444</td>
      <td>0.387609</td>
      <td>0.390823</td>
      <td>0.391946</td>
      <td>0.395230</td>
    </tr>
    <tr>
      <th>59</th>
      <td>sub-PDa073</td>
      <td>1160</td>
      <td>1160</td>
      <td>None</td>
      <td>None</td>
      <td>9</td>
      <td>2019-09-13 00:00:00</td>
      <td>None</td>
      <td>Y</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa073</td>
      <td>0.380700</td>
      <td>0.381698</td>
      <td>0.382503</td>
      <td>0.385815</td>
      <td>0.386705</td>
      <td>0.389182</td>
    </tr>
    <tr>
      <th>62</th>
      <td>sub-PDa079</td>
      <td>1220</td>
      <td>1220</td>
      <td>None</td>
      <td>None</td>
      <td>12</td>
      <td>2019-10-21 00:00:00</td>
      <td>None</td>
      <td>Y</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa079</td>
      <td>0.387683</td>
      <td>0.388123</td>
      <td>0.388603</td>
      <td>0.392015</td>
      <td>0.393306</td>
      <td>0.396756</td>
    </tr>
    <tr>
      <th>63</th>
      <td>sub-PDa080</td>
      <td>1240</td>
      <td>1240</td>
      <td>None</td>
      <td>None</td>
      <td>13</td>
      <td>2019-10-28 00:00:00</td>
      <td>None</td>
      <td>Y</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa080</td>
      <td>0.389607</td>
      <td>0.390514</td>
      <td>0.391020</td>
      <td>0.394144</td>
      <td>0.395031</td>
      <td>0.397516</td>
    </tr>
    <tr>
      <th>66</th>
      <td>sub-PDa084</td>
      <td>1300</td>
      <td>1300</td>
      <td>None</td>
      <td>None</td>
      <td>65</td>
      <td>2022-10-28 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa084</td>
      <td>0.378829</td>
      <td>0.379428</td>
      <td>0.380395</td>
      <td>0.384311</td>
      <td>0.385466</td>
      <td>0.388346</td>
    </tr>
    <tr>
      <th>67</th>
      <td>sub-PDa087</td>
      <td>1320</td>
      <td>1320</td>
      <td>None</td>
      <td>None</td>
      <td>16</td>
      <td>2020-02-03 00:00:00</td>
      <td>None</td>
      <td>Y</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa087</td>
      <td>0.395734</td>
      <td>0.396099</td>
      <td>0.396152</td>
      <td>0.398938</td>
      <td>0.399908</td>
      <td>0.402447</td>
    </tr>
    <tr>
      <th>68</th>
      <td>sub-PDa088</td>
      <td>1340</td>
      <td>1340</td>
      <td>None</td>
      <td>None</td>
      <td>17</td>
      <td>2020-05-18 00:00:00</td>
      <td>None</td>
      <td>Y</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa088</td>
      <td>0.379993</td>
      <td>0.378550</td>
      <td>0.377436</td>
      <td>0.377460</td>
      <td>0.377875</td>
      <td>0.379397</td>
    </tr>
    <tr>
      <th>69</th>
      <td>sub-PDa089</td>
      <td>1360</td>
      <td>1360</td>
      <td>None</td>
      <td>None</td>
      <td>18</td>
      <td>2020-05-26 00:00:00</td>
      <td>None</td>
      <td>Y</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa089</td>
      <td>0.377470</td>
      <td>0.378094</td>
      <td>0.378990</td>
      <td>0.382431</td>
      <td>0.383546</td>
      <td>0.386593</td>
    </tr>
    <tr>
      <th>74</th>
      <td>sub-PDa094</td>
      <td>1460</td>
      <td>1460</td>
      <td>None</td>
      <td>None</td>
      <td>22</td>
      <td>2020-10-02 00:00:00</td>
      <td>None</td>
      <td>Y</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa094</td>
      <td>0.368605</td>
      <td>0.368548</td>
      <td>0.368718</td>
      <td>0.371044</td>
      <td>0.371898</td>
      <td>0.374604</td>
    </tr>
    <tr>
      <th>78</th>
      <td>sub-PDa099</td>
      <td>1540</td>
      <td>1540</td>
      <td>None</td>
      <td>None</td>
      <td>66</td>
      <td>2022-11-01 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa099</td>
      <td>0.363852</td>
      <td>0.364175</td>
      <td>0.364833</td>
      <td>0.368478</td>
      <td>0.369516</td>
      <td>0.372138</td>
    </tr>
    <tr>
      <th>79</th>
      <td>sub-PDa101</td>
      <td>1560</td>
      <td>1560</td>
      <td>PDpm014</td>
      <td>None</td>
      <td>30</td>
      <td>2021-03-22 00:00:00</td>
      <td>2024-02-08</td>
      <td>Y</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa101</td>
      <td>0.374732</td>
      <td>0.375643</td>
      <td>0.377776</td>
      <td>0.383151</td>
      <td>0.384435</td>
      <td>0.387570</td>
    </tr>
    <tr>
      <th>80</th>
      <td>sub-PDa102</td>
      <td>1580</td>
      <td>1580</td>
      <td>None</td>
      <td>None</td>
      <td>28</td>
      <td>2021-04-19 00:00:00</td>
      <td>None</td>
      <td>Y</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa102</td>
      <td>0.364242</td>
      <td>0.364899</td>
      <td>0.365688</td>
      <td>0.369247</td>
      <td>0.370385</td>
      <td>0.373273</td>
    </tr>
    <tr>
      <th>81</th>
      <td>sub-PDa103</td>
      <td>1600</td>
      <td>1600</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2020-11-04 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa103</td>
      <td>0.384445</td>
      <td>0.385433</td>
      <td>0.386262</td>
      <td>0.389958</td>
      <td>0.391005</td>
      <td>0.393792</td>
    </tr>
    <tr>
      <th>85</th>
      <td>sub-PDa110</td>
      <td>1680</td>
      <td>1680</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2021-08-16 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa110</td>
      <td>0.381053</td>
      <td>0.381024</td>
      <td>0.381132</td>
      <td>0.383588</td>
      <td>0.384289</td>
      <td>0.386177</td>
    </tr>
    <tr>
      <th>90</th>
      <td>sub-PDa118</td>
      <td>1780</td>
      <td>1780</td>
      <td>None</td>
      <td>None</td>
      <td>33</td>
      <td>2021-11-15 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa118</td>
      <td>0.381077</td>
      <td>0.381560</td>
      <td>0.382154</td>
      <td>0.384482</td>
      <td>0.385340</td>
      <td>0.387756</td>
    </tr>
    <tr>
      <th>91</th>
      <td>sub-PDa119</td>
      <td>1800</td>
      <td>1800</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2021-11-25 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa119</td>
      <td>0.382647</td>
      <td>0.382889</td>
      <td>0.382337</td>
      <td>0.382604</td>
      <td>0.383061</td>
      <td>0.384549</td>
    </tr>
    <tr>
      <th>92</th>
      <td>sub-PDa120</td>
      <td>1820</td>
      <td>1820</td>
      <td>None</td>
      <td>None</td>
      <td>34</td>
      <td>2021-11-29 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa120</td>
      <td>0.376848</td>
      <td>0.377232</td>
      <td>0.377587</td>
      <td>0.380145</td>
      <td>0.381076</td>
      <td>0.383903</td>
    </tr>
    <tr>
      <th>95</th>
      <td>sub-PDa123</td>
      <td>1880</td>
      <td>1880</td>
      <td>PDpm008</td>
      <td>None</td>
      <td>36</td>
      <td>2021-12-22 00:00:00</td>
      <td>2023-11-13</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa123</td>
      <td>0.321705</td>
      <td>0.322762</td>
      <td>0.324191</td>
      <td>0.327777</td>
      <td>0.328689</td>
      <td>0.331026</td>
    </tr>
    <tr>
      <th>97</th>
      <td>sub-PDa125</td>
      <td>1920</td>
      <td>1920</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2016-03-16 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>73.666667</td>
      <td>67.0</td>
      <td>107.000000</td>
      <td>sub-PDa125</td>
      <td>0.388018</td>
      <td>0.388068</td>
      <td>0.389326</td>
      <td>0.396204</td>
      <td>0.397925</td>
      <td>0.401610</td>
    </tr>
    <tr>
      <th>99</th>
      <td>sub-PDa127</td>
      <td>1960</td>
      <td>1960</td>
      <td>None</td>
      <td>None</td>
      <td>68</td>
      <td>2022-11-03 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa127</td>
      <td>0.382083</td>
      <td>0.382841</td>
      <td>0.383729</td>
      <td>0.387839</td>
      <td>0.388998</td>
      <td>0.392097</td>
    </tr>
    <tr>
      <th>103</th>
      <td>sub-PDa133</td>
      <td>2040</td>
      <td>2040</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2022-04-24 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa133</td>
      <td>0.369084</td>
      <td>0.368932</td>
      <td>0.368544</td>
      <td>0.368805</td>
      <td>0.369070</td>
      <td>0.370268</td>
    </tr>
    <tr>
      <th>106</th>
      <td>sub-PDa136</td>
      <td>2100</td>
      <td>2100</td>
      <td>None</td>
      <td>None</td>
      <td>40.0</td>
      <td>2021-12-27 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa136</td>
      <td>0.389064</td>
      <td>0.389179</td>
      <td>0.389839</td>
      <td>0.393760</td>
      <td>0.394955</td>
      <td>0.397903</td>
    </tr>
    <tr>
      <th>107</th>
      <td>sub-PDa137</td>
      <td>2120</td>
      <td>2120</td>
      <td>PDpm006</td>
      <td>PDpf003</td>
      <td>41.0</td>
      <td>2022-05-02 00:00:00</td>
      <td>2023-10-17</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa137</td>
      <td>0.373622</td>
      <td>0.374482</td>
      <td>0.375649</td>
      <td>0.379652</td>
      <td>0.380779</td>
      <td>0.383526</td>
    </tr>
    <tr>
      <th>109</th>
      <td>sub-PDa139</td>
      <td>2160</td>
      <td>2160</td>
      <td>None</td>
      <td>PDpf005</td>
      <td>None</td>
      <td>2022-01-13 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa139</td>
      <td>0.380131</td>
      <td>0.380070</td>
      <td>0.379686</td>
      <td>0.380397</td>
      <td>0.380863</td>
      <td>0.382544</td>
    </tr>
    <tr>
      <th>117</th>
      <td>sub-PDa148</td>
      <td>2320</td>
      <td>2320</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2022-08-14 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa148</td>
      <td>0.371439</td>
      <td>0.374186</td>
      <td>0.376818</td>
      <td>0.383157</td>
      <td>0.384638</td>
      <td>0.388216</td>
    </tr>
    <tr>
      <th>118</th>
      <td>sub-PDa149</td>
      <td>2340</td>
      <td>2340</td>
      <td>None</td>
      <td>None</td>
      <td>69.0</td>
      <td>2022-10-31 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa149</td>
      <td>0.381863</td>
      <td>0.383032</td>
      <td>0.383824</td>
      <td>0.387191</td>
      <td>0.388228</td>
      <td>0.390860</td>
    </tr>
    <tr>
      <th>120</th>
      <td>sub-PDa151</td>
      <td>2380</td>
      <td>2380</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2021-08-03 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa151</td>
      <td>0.360658</td>
      <td>0.361201</td>
      <td>0.361764</td>
      <td>0.364362</td>
      <td>0.365132</td>
      <td>0.367130</td>
    </tr>
    <tr>
      <th>122</th>
      <td>sub-PDa174</td>
      <td>2420</td>
      <td>2420</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2021-03-26 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa174</td>
      <td>0.386962</td>
      <td>0.387727</td>
      <td>0.387995</td>
      <td>0.390438</td>
      <td>0.391296</td>
      <td>0.393506</td>
    </tr>
    <tr>
      <th>123</th>
      <td>sub-PDa189</td>
      <td>2440</td>
      <td>2440</td>
      <td>None</td>
      <td>None</td>
      <td>81.0</td>
      <td>2019-04-24 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa189</td>
      <td>0.343995</td>
      <td>0.344992</td>
      <td>0.347722</td>
      <td>0.354061</td>
      <td>0.355405</td>
      <td>0.358651</td>
    </tr>
    <tr>
      <th>125</th>
      <td>sub-PDa202</td>
      <td>2480</td>
      <td>2480</td>
      <td>PDpm012</td>
      <td>None</td>
      <td>64.0</td>
      <td>2021-03-02 00:00:00</td>
      <td>2024-03-17</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa202</td>
      <td>0.398778</td>
      <td>0.399460</td>
      <td>0.400165</td>
      <td>0.404211</td>
      <td>0.405321</td>
      <td>0.408036</td>
    </tr>
    <tr>
      <th>126</th>
      <td>sub-PDa213</td>
      <td>2500</td>
      <td>2500</td>
      <td>None</td>
      <td>None</td>
      <td>83.0</td>
      <td>4/14/21</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa213</td>
      <td>0.378079</td>
      <td>0.378654</td>
      <td>0.379194</td>
      <td>0.381404</td>
      <td>0.382009</td>
      <td>0.383835</td>
    </tr>
    <tr>
      <th>127</th>
      <td>sub-PDa287</td>
      <td>2520</td>
      <td>2520</td>
      <td>None</td>
      <td>None</td>
      <td>62.0</td>
      <td>2021-05-26 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa287</td>
      <td>0.377452</td>
      <td>0.378102</td>
      <td>0.378820</td>
      <td>0.382871</td>
      <td>0.384086</td>
      <td>0.387251</td>
    </tr>
    <tr>
      <th>128</th>
      <td>sub-PDa290</td>
      <td>2540</td>
      <td>2540</td>
      <td>None</td>
      <td>None</td>
      <td>84.0</td>
      <td>2020-09-01 00:00:00</td>
      <td>2024-09-10</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa290</td>
      <td>0.384932</td>
      <td>0.385249</td>
      <td>0.386155</td>
      <td>0.389506</td>
      <td>0.390686</td>
      <td>0.393917</td>
    </tr>
    <tr>
      <th>134</th>
      <td>sub-PDa373</td>
      <td>2660</td>
      <td>2660</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2022-10-21 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa373</td>
      <td>0.381236</td>
      <td>0.382127</td>
      <td>0.383884</td>
      <td>0.388347</td>
      <td>0.389645</td>
      <td>0.393241</td>
    </tr>
    <tr>
      <th>135</th>
      <td>sub-PDa374</td>
      <td>2680</td>
      <td>2680</td>
      <td>None</td>
      <td>None</td>
      <td>71.0</td>
      <td>2022-10-25 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa374</td>
      <td>0.385062</td>
      <td>0.385418</td>
      <td>0.386201</td>
      <td>0.391018</td>
      <td>0.392300</td>
      <td>0.395499</td>
    </tr>
    <tr>
      <th>136</th>
      <td>sub-PDa375</td>
      <td>2700</td>
      <td>2700</td>
      <td>None</td>
      <td>None</td>
      <td>72.0</td>
      <td>2022-11-07 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa375</td>
      <td>0.379015</td>
      <td>0.378816</td>
      <td>0.378751</td>
      <td>0.382212</td>
      <td>0.383296</td>
      <td>0.386022</td>
    </tr>
    <tr>
      <th>138</th>
      <td>sub-PDa382</td>
      <td>2740</td>
      <td>2740</td>
      <td>None</td>
      <td>None</td>
      <td>75.0</td>
      <td>2023-01-22 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa382</td>
      <td>0.387188</td>
      <td>0.387684</td>
      <td>0.388535</td>
      <td>0.393371</td>
      <td>0.394713</td>
      <td>0.398079</td>
    </tr>
    <tr>
      <th>140</th>
      <td>sub-PDa394</td>
      <td>2780</td>
      <td>2780</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2023-07-24 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa394</td>
      <td>0.376015</td>
      <td>0.377029</td>
      <td>0.378156</td>
      <td>0.382243</td>
      <td>0.383329</td>
      <td>0.386158</td>
    </tr>
    <tr>
      <th>143</th>
      <td>sub-PDa402</td>
      <td>2840</td>
      <td>2840</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2023-05-12 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa402</td>
      <td>0.357774</td>
      <td>0.357437</td>
      <td>0.357590</td>
      <td>0.358806</td>
      <td>0.359423</td>
      <td>0.361289</td>
    </tr>
    <tr>
      <th>144</th>
      <td>sub-PDa409</td>
      <td>2860</td>
      <td>2860</td>
      <td>None</td>
      <td>None</td>
      <td>90.0</td>
      <td>2023-07-24 00:00:00</td>
      <td>2024-04-23</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa409</td>
      <td>0.391404</td>
      <td>0.392053</td>
      <td>0.392887</td>
      <td>0.397328</td>
      <td>0.398567</td>
      <td>0.401733</td>
    </tr>
    <tr>
      <th>145</th>
      <td>sub-PDa411</td>
      <td>2880</td>
      <td>2880</td>
      <td>None</td>
      <td>None</td>
      <td>91.0</td>
      <td>2023-09-01 00:00:00</td>
      <td>None</td>
      <td>N</td>
      <td>Y</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub-PDa411</td>
      <td>0.394639</td>
      <td>0.395941</td>
      <td>0.396622</td>
      <td>0.399024</td>
      <td>0.399697</td>
      <td>0.401606</td>
    </tr>
  </tbody>
</table>
<p>57 rows × 342 columns</p>
</div>



# Check number of patient values present


```python
# Import necessary library
import pandas as pd

# Load the uploaded spreadsheet using relative path
metadata_dir = os.path.join('..', 'data', 'metadata')
file_path = os.path.join(metadata_dir, 'Deidentified_master_spreadsheet_01.07.25_SD.xlsx')
spreadsheet = pd.ExcelFile(file_path)

# Load the specific sheet containing the clinical data
df = spreadsheet.parse('Retro Clin')

# Define the columns of interest including LEDDpost and MDSUPDRS subscore columns for Brady, Axial, Tremor
columns_of_interest = ['PTID_Retro_Clin', 'LEDDpost1_Delta', 'LEDDpost2_Delta', 'LEDDpost3_Delta', 
                           'MDSUPDRSIIIpre_Percent_TOTAL_V',
    'MDSUPDRSIIIpre_Percent_BRADY',
    'MDSUPDRSIIIpre_Percent_TREMOR',
    'MDSUPDRSIIIpre_Percent_AXIAL']

# Extract subject IDs from user's list
subject_ids_from_list = [
    'PDa001', 'PDa002', 'PDa003', 'PDa005', 'PDa006', 'PDa007', 'PDa008', 'PDa009', 'PDa010',
    'PDa011', 'PDa012', 'PDa013', 'PDa014', 'PDa016', 'PDa017', 'PDa018', 'PDa019', 'PDa020',
    'PDa021', 'PDa022', 'PDa023', 'PDa024', 'PDa025', 'PDa026', 'PDa027', 'PDa028', 'PDa029',
    'PDa030', 'PDa031', 'PDa032', 'PDa034', 'PDa035', 'PDa036', 'PDa037', 'PDa038', 'PDa039',
    'PDa040', 'PDa042', 'PDa043', 'PDa044', 'PDa045', 'PDa046', 'PDa047', 'PDa050', 'PDa051',
    'PDa054', 'PDa055', 'PDa056', 'PDa057', 'PDa058', 'PDa060', 'PDa061', 'PDa062', 'PDa063',
    'PDa064', 'PDa065', 'PDa066', 'PDa070', 'PDa071', 'PDa073', 'PDa077', 'PDa078', 'PDa079',
    'PDa080', 'PDa081', 'PDa082', 'PDa084', 'PDa087', 'PDa088', 'PDa089', 'PDa090', 'PDa091',
    'PDa092', 'PDa093', 'PDa094', 'PDa096', 'PDa097', 'PDa098', 'PDa099', 'PDa101', 'PDa102',
    'PDa103', 'PDa104', 'PDa108', 'PDa109', 'PDa110', 'PDa112', 'PDa115', 'PDa116', 'PDa117',
    'PDa118', 'PDa119', 'PDa120', 'PDa121', 'PDa122', 'PDa123', 'PDa124', 'PDa125', 'PDa126',
    'PDa127', 'PDa128', 'PDa130', 'PDa132', 'PDa133', 'PDa134', 'PDa135', 'PDa136', 'PDa137',
    'PDa138', 'PDa139', 'PDa140', 'PDa141', 'PDa142', 'PDa143', 'PDa144', 'PDa146', 'PDa147',
    'PDa148', 'PDa149', 'PDa150', 'PDa151', 'PDa165', 'PDa174', 'PDa189', 'PDa192', 'PDa202',
    'PDa213', 'PDa287', 'PDa290', 'PDa299', 'PDa347', 'PDa365', 'PDa371', 'PDa372', 'PDa373',
    'PDa374', 'PDa375', 'PDa376', 'PDa382', 'PDa391', 'PDa394', 'PDa399', 'PDa401', 'PDa402',
    'PDa409', 'PDa411'
]

# Filter the data for the selected columns and subject IDs
filtered_led_delta_data = df[df['PTID_Retro_Clin'].isin(subject_ids_from_list)][columns_of_interest]

# Count the number of non-NA values for each of the LEDDpost columns
value_counts = filtered_led_delta_data[['LEDDpost1_Delta', 'LEDDpost2_Delta', 'LEDDpost3_Delta']].notna().sum()

# Check the total number of patients
total_patients = df['PTID_Retro_Clin'].nunique()

# Check the number of patients with non-NA values for the MDSUPDRS total V score
patients_with_mdsupdrs_total_v = filtered_led_delta_data['MDSUPDRSIIIpre_Percent_TOTAL_V'].notna().sum()

# Get the list of patients missing MDSUPDRS total V values
missing_mdsupdrs_patients = filtered_led_delta_data[filtered_led_delta_data['MDSUPDRSIIIpre_Percent_TOTAL_V'].isna()]['PTID_Retro_Clin'].tolist()

# Output the results
filtered_led_delta_data, value_counts, total_patients, patients_with_mdsupdrs_total_v, missing_mdsupdrs_patients
```




    (    PTID_Retro_Clin LEDDpost1_Delta LEDDpost2_Delta LEDDpost3_Delta  \
     0            PDa001       41.025641       -6.410256       37.076923   
     1            PDa002       56.561086       50.980392       62.292609   
     2            PDa003       81.818182       78.787879        69.69697   
     4            PDa005      -25.641026               0               0   
     5            PDa006               0             NaN             NaN   
     ..              ...             ...             ...             ...   
     398          PDa399           34.28             NaN           46.55   
     400          PDa401           80.91             NaN             NaN   
     401          PDa402        6.317119        3.600758       74.794694   
     408          PDa409        2.256318               0       14.620939   
     410          PDa411       13.043478        -3.26087       13.043478   
     
         MDSUPDRSIIIpre_Percent_TOTAL_V MDSUPDRSIIIpre_Percent_BRADY  \
     0                         60.97561                    52.631579   
     1                              NaN                          NaN   
     2                              NaN                          NaN   
     4                               75                    66.666667   
     5                             12.5                            8   
     ..                             ...                          ...   
     398                      30.769231                           50   
     400                            NaN                          NaN   
     401                      61.538462                    55.555556   
     408                         53.125                    45.454545   
     410                      53.333333                    47.619048   
     
         MDSUPDRSIIIpre_Percent_TREMOR MDSUPDRSIIIpre_Percent_AXIAL  
     0                       61.538462                    64.285714  
     1                             NaN                          NaN  
     2                             NaN                          NaN  
     4                             100                    90.909091  
     5                             100                    16.666667  
     ..                            ...                          ...  
     398                            15                          NaN  
     400                           NaN                          NaN  
     401                            60                          100  
     408                            80                          NaN  
     410                     71.428571                          NaN  
     
     [146 rows x 8 columns],
     LEDDpost1_Delta    129
     LEDDpost2_Delta     97
     LEDDpost3_Delta     91
     dtype: int64,
     531,
     np.int64(121),
     ['PDa002',
      'PDa003',
      'PDa009',
      'PDa010',
      'PDa016',
      'PDa018',
      'PDa019',
      'PDa026',
      'PDa027',
      'PDa030',
      'PDa035',
      'PDa036',
      'PDa042',
      'PDa047',
      'PDa051',
      'PDa063',
      'PDa070',
      'PDa081',
      'PDa092',
      'PDa124',
      'PDa125',
      'PDa133',
      'PDa347',
      'PDa394',
      'PDa401'])




```python
df['LEDDpre_Date'] = pd.to_datetime(df['LEDDpre_Date'], format='%m/%d/%y', errors='coerce')
df['LEDDpost1_Date'] = pd.to_datetime(df['LEDDpost1_Date'], format='%m/%d/%y', errors='coerce')

# Drop rows where either date is NaT (not a date)
df = df.dropna(subset=['LEDDpre_Date', 'LEDDpost1_Date'])

# Calculate absolute differences in days
df['Date_Difference'] = (df['LEDDpost1_Date'] - df['LEDDpre_Date']).abs().dt.days

# Find the top 10 smallest differences
top_10_smallest_distances = df.nsmallest(10, 'Date_Difference')[['PTID_Retro_Clin', 'Date_Difference']]

# Display the result
print(top_10_smallest_distances)
highest_age = patient_df['Age_DBS_On'].max()
print(highest_age)
lowest_age = patient_df['Age_DBS_On'].min()
print(lowest_age)
```

        PTID_Retro_Clin  Date_Difference
    46           PDa047               10
    89           PDa090               12
    29           PDa030               15
    69           PDa070               40
    488          PDa489               42
    451          PDa452               43
    489          PDa490               43
    0            PDa001               44
    490          PDa491               45
    416          PDa417               49
    86.43
    28.96
       Min_Year  Max_Year
    0       NaN       NaN



```python
# Import necessary library
import pandas as pd

# Load the uploaded spreadsheet using relative path
metadata_dir = os.path.join('..', 'data', 'metadata')
file_path = os.path.join(metadata_dir, 'Deidentified_master_spreadsheet_01.07.25_SD.xlsx')
spreadsheet = pd.ExcelFile(file_path)

# Load the specific sheet containing the clinical data
df = spreadsheet.parse('Retro Clin')

# Define the columns of interest including LEDDpost and MDSUPDRS subscore columns for Brady, Axial, Tremor
columns_of_interest = ['PTID_Retro_Clin', 'LEDDpost1_Delta', 'LEDDpost2_Delta', 'LEDDpost3_Delta', 
                           'MDSUPDRSIIIpre_Percent_TOTAL_V',
                           'MDSUPDRSIIIpre_Percent_BRADY',
                           'MDSUPDRSIIIpre_Percent_TREMOR',
                           'MDSUPDRSIIIpre_Percent_AXIAL']

# Extract subject IDs from user's list
subject_ids_from_list = [
    'PDa001', 'PDa002', 'PDa003', 'PDa005', 'PDa006', 'PDa007', 'PDa008', 'PDa009', 'PDa010',
    'PDa011', 'PDa012', 'PDa013', 'PDa014', 'PDa016', 'PDa017', 'PDa018', 'PDa019', 'PDa020',
    #... (rest of the subject IDs)
    'PDa409', 'PDa411'
]

# Filter the data for the selected columns and subject IDs
filtered_led_delta_data = df[df['PTID_Retro_Clin'].isin(subject_ids_from_list)][columns_of_interest]

# Count the number of non-NA values for each of the LEDDpost columns
value_counts = filtered_led_delta_data[['LEDDpost1_Delta', 'LEDDpost2_Delta', 'LEDDpost3_Delta']].notna().sum()

# Check the total number of patients
total_patients = df['PTID_Retro_Clin'].nunique()

# Check the number of patients with non-NA values for the MDSUPDRS total V score
patients_with_mdsupdrs_total_v = df['MDSUPDRSIIIpre_Percent_TOTAL_V'].notna().sum()

# Display filtered dataframe for LEDDpost columns along with MDSUPDRS subscores for Brady, Axial, and Tremor
filtered_led_delta_data, value_counts, total_patients, patients_with_mdsupdrs_total_v

```




    (    PTID_Retro_Clin LEDDpost1_Delta LEDDpost2_Delta LEDDpost3_Delta  \
     0            PDa001       41.025641       -6.410256       37.076923   
     1            PDa002       56.561086       50.980392       62.292609   
     2            PDa003       81.818182       78.787879        69.69697   
     4            PDa005      -25.641026               0               0   
     5            PDa006               0             NaN             NaN   
     6            PDa007        9.225092             100             NaN   
     7            PDa008               0             NaN             NaN   
     8            PDa009       70.731707       63.414634        89.02439   
     9            PDa010             NaN       26.315789       10.526316   
     10           PDa011        5.263158        5.263158        5.263158   
     11           PDa012        5.743425       73.161567        91.94847   
     12           PDa013               0       30.442324       30.442324   
     13           PDa014               0             NaN             NaN   
     15           PDa016              90             100              70   
     16           PDa017               0             NaN             NaN   
     17           PDa018               0             NaN             NaN   
     18           PDa019               0             NaN             NaN   
     19           PDa020              40             -20             NaN   
     408          PDa409        2.256318               0       14.620939   
     410          PDa411       13.043478        -3.26087       13.043478   
     
         MDSUPDRSIIIpre_Percent_TOTAL_V MDSUPDRSIIIpre_Percent_BRADY  \
     0                         60.97561                    52.631579   
     1                              NaN                          NaN   
     2                              NaN                          NaN   
     4                               75                    66.666667   
     5                             12.5                            8   
     6                         64.40678                    57.142857   
     7                        59.459459                    54.166667   
     8                              NaN                          NaN   
     9                              NaN                          NaN   
     10                              48                           50   
     11                       45.454545                    31.818182   
     12                           43.75                    28.571429   
     13                       44.897959                    16.666667   
     15                             NaN                          NaN   
     16                       14.285714                           10   
     17                             NaN                          NaN   
     18                             NaN                          NaN   
     19                        54.83871                           40   
     408                         53.125                    45.454545   
     410                      53.333333                    47.619048   
     
         MDSUPDRSIIIpre_Percent_TREMOR MDSUPDRSIIIpre_Percent_AXIAL  
     0                       61.538462                    64.285714  
     1                             NaN                          NaN  
     2                             NaN                          NaN  
     4                             100                    90.909091  
     5                             100                    16.666667  
     6                       82.352941                    55.555556  
     7                             100                    33.333333  
     8                             NaN                          NaN  
     9                             NaN                          NaN  
     10                            NaN                           40  
     11                      77.777778                           50  
     12                      71.428571                           80  
     13                             68                        16.67  
     15                            NaN                          NaN  
     16                      27.272727                            0  
     17                            NaN                          NaN  
     18                            NaN                          NaN  
     19                            100                    46.153846  
     408                            80                          NaN  
     410                     71.428571                          NaN  ,
     LEDDpost1_Delta    19
     LEDDpost2_Delta    14
     LEDDpost3_Delta    12
     dtype: int64,
     531,
     np.int64(351))




```python
num_values = patient_df["MDSUPDRSIIIpre_Percent_TOTAL"].notna().sum()

print(f"Number of values in 'MDSUPDRSIIIpre_Percent_TOTAL': {num_values}")
```

    Number of values in 'MDSUPDRSIIIpre_Percent_TOTAL': 83


# III. Regression Analyses

### Create function for plotting and finding R correlation coefficient between two spreadsheet variables, staggering by tract.

The 'covariates' parameter takes in a list of column names that will be used as covariates for multiple regression analysis.

The 'stagger_column' parameter takes in a column name, and then assigns a different color to plot unique values within that column.


```python
def plot_correlation(data, x_vars, y_var, covariates=None, stagger_column=None, x_label=None, y_label=None, add_best_fit=False):
    # Ensure no grid style and set color palette for staggered plots
    sns.set(style="whitegrid", rc={'axes.grid': False})
    color_palette = ["black", "white"]  # Define more colors if you have more categories
    
    # Initialize dataframe to store tract name, r value, and p value
    correlation_df = pd.DataFrame(columns=['TractID', 'X_Var', 'R_Value', 'P_Value'])
    
    # Get unique tractIDs
    tract_ids = sorted(data['tractID'].unique())
    
    # Create subplots, one per tractID, arranged vertically
    subplot_height = 8  # Increase this value to give more space to each subplot
    total_fig_height = subplot_height * len(tract_ids)
    
    # Create subplots, one per tractID, arranged vertically
    fig, axes = plt.subplots(len(tract_ids), 1, figsize=(8, total_fig_height), squeeze=False)
    
    # Flatten the axes array for easy iteration if we have only one column
    axes = axes.flatten()
    
    # Iterate over each tractID
    for i, tract_id in enumerate(tract_ids):
        ax = axes[i]
        
        # Filter data for the current tractID and create a copy to avoid SettingWithCopyWarning
        tract_data = data[data['tractID'] == tract_id].copy()
        
        # If a covariate is specified, adjust the y variable
        if covariates:
            covariate_formula = ' + '.join(covariates)
            model_formula = f"{y_var} ~ {covariate_formula}"
            model = ols(model_formula, data=tract_data).fit()
            tract_data[y_var] = model.resid
        
        text_pos_y = -0.1  # Start position for the text, adjust as needed
        
        # Plot each x variable with its corresponding color
        for k, x_var in enumerate(x_vars):
            # Remove missing data
            plot_data = tract_data.dropna(subset=[x_var, y_var])

            if len(plot_data) >= 2:
                if stagger_column:
                    # Plot each stagger column value with a different color
                    stagger_values = plot_data[stagger_column].unique()
                    for idx, value in enumerate(stagger_values):
                        subset = plot_data[plot_data[stagger_column] == value]
                        sns.scatterplot(x=x_var, y=y_var, data=subset, ax=ax, 
                                        color=color_palette[idx % len(color_palette)], edgecolor = "black", label=str(value))
                else:
                    # No stagger column, plot normally
                    sns.scatterplot(x=x_var, y=y_var, data=plot_data, ax=ax, 
                                    color=color_palette[k % len(color_palette)], edgecolor = "black", label=x_var)

                # Calculate Pearson correlation coefficient and p-value
                r, p = pearsonr(plot_data[x_var], plot_data[y_var])
                
                # Append results to dataframe
                new_row = pd.DataFrame({'TractID': [tract_id], 'X_Var': [x_var], 'R_Value': [r], 'P_Value': [p]})
                correlation_df = pd.concat([correlation_df, new_row], ignore_index=True)
                
                # Display r and p-value below the plot
                # ax.text(0.5, text_pos_y, f'{x_var}: r={r:.2f}, p={p:.2g}', 
            #tempformelanie
                ax.text(0.5, text_pos_y, f'r={r:.2f}, p={p:.2g}', 

                        transform=ax.transAxes, horizontalalignment='center', verticalalignment='top', 
                        color=color_palette[k % len(color_palette)])
                
                # Adjust vertical position for the next annotation
                text_pos_y -= 0.05
                
                if add_best_fit:
                    sns.regplot(x=x_var, y=y_var, data=plot_data, ax=ax, scatter=False, 
                                color=color_palette[k % len(color_palette)], line_kws={"color": "red"})
        # Set axis labels
        #if x_label:
            ax.set_xlabel(f"x_var: {x_var}", fontsize = 4)
       # if y_label:
            ax.set_ylabel('∆LEDD', fontsize = 26)
            ax.tick_params(axis='both', labelsize=24)
        # Set title for each subplot
            ax.set_title(f"TractID: {tract_id}", fontsize = 18)
        #tempformelanie
       # ax.set_title(f"LEDD Changes against MDS-UPDRS with Age, Sex, and White Matter Tract Covariates", fontsize = 18)
        
        # Display legend below the axis
        if stagger_column or len(x_vars) > 1:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                      ncol=3, frameon=False)

    # Adjust layout to make space for legend below the plots
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save the figure with the dynamic file name
    x_vars_str = "_".join(x_vars)
    y_var_str = y_var
    filename = f"corr_{x_vars_str}"
    if covariates:
        filename += f"_{covariates}"
    if stagger_column:
        filename += f"_{stagger_column}"
    filename += f"_{y_var_str}.png"
    plt.savefig(filename, bbox_inches='tight')
    correlation_df.to_csv(f"stats_{filename}.csv")

    plt.show()
    
    # Return the dataframe with correlation results
    return correlation_df

```

### Create function for plotting and finding R correlation coefficient between two spreadsheet variables, without staggering by tract.


```python
def plot_correlation_alltracts(
    data,
    x_vars,
    y_var,
    covariates=None,
    stagger_column=None,
    title=None,
    x_label=None,
    y_label=None,
    add_best_fit=False,
    save_filename=None,       # Optional: Allow custom filename
    show_legend=False,        # Control legend display
    show_stats=True           # Control statistics display
):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from statsmodels.formula.api import ols
    import os
    import datetime

    # Validate input variables
    if not isinstance(x_vars, list):
        raise ValueError("x_vars should be a list of variable names.")
    if not isinstance(y_var, str):
        raise ValueError("y_var should be a single variable name as a string.")
    if covariates and not isinstance(covariates, list):
        raise ValueError("covariates should be a list of variable names or None.")

    # Set the Seaborn style without gridlines and high DPI for publication
    sns.set(style="white", rc={'axes.grid': False, 'figure.dpi': 300})  

    # Use Seaborn's default deep color palette
    color_palette = sns.color_palette("deep")

    # Define fixed square figure size
    fig, ax1 = plt.subplots(figsize=(8, 8))  # Square figure

    stats_texts = []
    for k, x_var in enumerate(x_vars):
        # Drop missing data in x_var, y_var, and covariates
        if covariates:
            required_cols = [x_var, y_var] + covariates
            plot_data = data.dropna(subset=required_cols).copy()
        else:
            plot_data = data.dropna(subset=[x_var, y_var]).copy()

        n = len(plot_data)

        if n >= 2:
            if covariates:
                # Regress y_var on covariates and get residuals for plotting
                y_formula = f"{y_var} ~ {' + '.join(covariates)}"
                y_model = ols(y_formula, data=plot_data).fit()
                plot_data['y_resid'] = y_model.resid

                # Regress x_var on covariates and get residuals for plotting
                x_formula = f"{x_var} ~ {' + '.join(covariates)}"
                x_model = ols(x_formula, data=plot_data).fit()
                plot_data['x_resid'] = x_model.resid

                # Use residuals for plotting
                x_plot = 'x_resid'
                y_plot = 'y_resid'

                # For statistical calculation, fit the full model
                full_formula = f"{y_var} ~ {x_var} + {' + '.join(covariates)}"
                full_model = ols(full_formula, data=plot_data).fit()
                r_squared = full_model.rsquared_adj
                p_value = full_model.pvalues[x_var]
            else:
                x_plot = x_var
                y_plot = y_var

                # Fit model without covariates
                full_model = ols(f"{y_var} ~ {x_var}", data=plot_data).fit()
                r_squared = full_model.rsquared_adj
                p_value = full_model.pvalues[x_var]

            # Determine label based on stagger_column and show_legend
            if stagger_column:
                label = plot_data[stagger_column].iloc[0] if show_legend else None
            else:
                label = x_var if show_legend else None

            # Plotting
            if stagger_column:
                stagger_values = plot_data[stagger_column].unique()
                for idx, value in enumerate(stagger_values):
                    subset = plot_data[plot_data[stagger_column] == value]
                    sns.scatterplot(
                        x=x_plot, y=y_plot, data=subset, ax=ax1, 
                        color=color_palette[idx % len(color_palette)], edgecolor="black", 
                        s=100, marker="o",
                        label=value if show_legend else None,
                        alpha=0.7
                    )
            else:
                sns.scatterplot(
                    x=x_plot, y=y_plot, data=plot_data, ax=ax1, 
                    color=color_palette[k % len(color_palette)], edgecolor="black", 
                    s=100, marker="o",
                    label=x_var if show_legend else None,
                    alpha=0.7
                )
            if add_best_fit:
                sns.regplot(
                    x=x_plot, y=y_plot, data=plot_data, ax=ax1, scatter=False, 
                    color="red", line_kws={"color": "red", "linewidth": 2}
                )

            # Add statistical information including the number of data points
            stats_texts.append(f'{x_var}: n={n}, $R^2$ = {r_squared:.2f}, p = {p_value:.3e}')
        else:
            print(f"Not enough data points for variable '{x_var}'. Skipping plot.")

    # Replace underscores with spaces for labels
    def format_label(label):
        return label.replace('_', ' ')

    # Format axis labels with bold fonts and remove gridlines
    if x_label:
        ax1.set_xlabel(x_label, fontsize=18, fontweight='bold', labelpad=10)
    else:
        if len(x_vars) == 1:
            ax1.set_xlabel(format_label(x_vars[0]), fontsize=18, fontweight='bold', labelpad=10)
        else:
            ax1.set_xlabel('Independent Variables', fontsize=18, fontweight='bold', labelpad=10)

    if y_label:
        ax1.set_ylabel(y_label, fontsize=18, fontweight='bold', labelpad=10)
    else:
        ax1.set_ylabel(format_label(y_var), fontsize=18, fontweight='bold', labelpad=10)

    ax1.tick_params(axis='both', labelsize=20, width=2)  # Make tick labels larger and more pronounced

    # Set default title
    if title:
        ax1.set_title(title, fontsize=20, fontweight='bold', pad=20)
    else:
        if covariates:
            covariate_labels = ', '.join([format_label(cov) for cov in covariates])
            ax1.set_title(f'Partial Regression Plot (Adjusted for {covariate_labels})', fontsize=20, fontweight='bold', pad=20)
        else:
            ax1.set_title('', fontsize=20, fontweight='bold', pad=20)

    # No gridlines, no major/minor grids
    ax1.grid(False)

    # Handle Legend
    if show_legend:
        # Place the legend inside the main plot area
        ax1.legend(
            title='Legend',
            fontsize=16,
            title_fontsize=18,
            loc='upper right',
            frameon=False
        )
    else:
        # Remove any existing legends
        if ax1.get_legend():
            ax1.get_legend().remove()

    # Add statistical information as text below the plot
    if show_stats:
        # Position the stats text below the plot within the figure
        stats_text = '\n'.join(stats_texts)
        plt.gcf().text(0.5, 0.02, stats_text, ha='center', fontsize=16, fontweight='bold')
    
    # Adjust layout to minimize white space
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for stats

    # Determine the current working directory
    current_dir = os.getcwd()

    # Construct the save filename if not provided
    if not save_filename:
        # Sanitize variable names by replacing spaces with underscores
        sanitized_y_var = y_var.replace(' ', '_')
        sanitized_x_vars = '_'.join([var.replace(' ', '_') for var in x_vars])
        if covariates:
            sanitized_covariates = '_'.join([cov.replace(' ', '_') for cov in covariates])
            filename = f"{sanitized_y_var}_vs_{sanitized_x_vars}_adjusted_for_{sanitized_covariates}.png"
        else:
            filename = f"{sanitized_y_var}_vs_{sanitized_x_vars}.png"
        save_filename = filename

    # Ensure the filename is safe
    save_filename = "".join([c for c in save_filename if c.isalpha() or c.isdigit() or c in (' ', '_', '-', '.')]).rstrip()

    # Construct the full save path
    save_path = os.path.join(current_dir, save_filename)

    # Prevent overwriting by appending a timestamp if file exists
    if os.path.exists(save_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(save_filename)
        save_filename = f"{base}_{timestamp}{ext}"
        save_path = os.path.join(current_dir, save_filename)

    # Save the figure as an image file in the current working directory
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')  # Added bbox_inches='tight'

    plt.show()

    print(f"Plot saved to: {save_path}")

```

### Create function that takes a list of strings representing covariates and finds the combination that results in the highest R^2 value for a given x and y


```python
import itertools
def find_best_covariates(data, x_vars, y_var, possible_covariates, max_covariates):
    best_r_squared = -1
    best_combination = []
    
    # Generate all possible combinations of the covariates up to the specified max_covariates
    for r in range(min(max_covariates, len(possible_covariates)) + 1):
        for combination in itertools.combinations(possible_covariates, r):
            covariates = list(combination)
            # Adjust y variable if covariates are specified
            if covariates:
                covariate_formula = ' + '.join(covariates)
                model_formula = f"{y_var} ~ {covariate_formula}"
                model = ols(model_formula, data=data).fit()
                data[y_var + '_adj'] = model.resid
                adjusted_y_var = y_var + '_adj'
            else:
                adjusted_y_var = y_var

            # Evaluate the model for each x_var with the current combination of covariates
            for x_var in x_vars:
                plot_data = data.dropna(subset=[x_var, adjusted_y_var])
                if len(plot_data) >= 2:
                    if covariates:
                        full_formula = f"{adjusted_y_var} ~ {x_var} + {' + '.join(covariates)}"
                    else:
                        full_formula = f"{adjusted_y_var} ~ {x_var}"
                    model = ols(full_formula, data=plot_data).fit()
                    r_squared = model.rsquared

                    # Check if this combination gives a better R²
                    if r_squared > best_r_squared:
                        best_r_squared = r_squared
                        best_combination = covariates
    
    return best_combination, best_r_squared
```

### Specify output directory


```python
# Define the relative path for the output directory for figures
output_directory = os.path.join('..', 'results', 'figures')
os.makedirs(output_directory, exist_ok=True) # Ensure the directory exists
# Note: os.chdir changes the working directory for the *entire script*.
# It's generally better practice to construct full paths for saving files
# rather than changing the working directory. We will adjust saving paths later.
# os.chdir(output_directory) # Avoid changing the working directory globally
```


```python
def process_target_lr(target_lr_series):
    """
    Process the 'Target_L_R' column to create a binary variable:
    - If it begins with 'STN', assign 1
    - If it begins with 'GPi', assign 0
    - Otherwise, assign NaN
    """
    def map_target(value):
        if isinstance(value, str):
            if value.startswith('STN'):
                return 1
            elif value.startswith('GPi'):
                return 0
        elif value == 1:
            return 1
        elif value == 0:
            return 0
        return np.nan

    return target_lr_series.apply(map_target)

def process_sex(sex_series):
    """
    Standardize the 'Sex' column to binary:
    - 'Male' -> 1
    - 'Female' -> 0
    - All others -> NaN
    """
    def map_sex(value):
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower.startswith('m') or 'male' in value_lower:
                return 1
            elif value_lower.startswith('f') or 'female' in value_lower:
                return 0
        return np.nan

    return sex_series.apply(map_sex)

def process_no_leads(no_leads_series):
    """
    Clean the 'No_Leads' column:
    - Convert 'BI' to 'Bi'
    - Keep only 'Bi', 'R', 'L'
    - Assign numerical values: 'Bi' = 2, 'R' = 1, 'L' = 0
    - Remove any entries not in ['Bi', 'R', 'L']
    """
    def map_no_leads(value):
        if isinstance(value, str):
            value = value.strip().upper()
            if value == 'BI':
                return 2
            elif value == 'R':
                return 1
            elif value == 'L':
                return 1
        return 0

    return no_leads_series.apply(map_no_leads)

def simplify_feature_name(feature_name):
    """
    Simplify feature names for better readability in plots.
    """
    # Example: Remove prefix or suffix if necessary
    return feature_name.replace('Sex_binary', 'Sex') \
                       .replace('No_Leads_numeric', 'No_Leads') \
                       .replace('tractID_', '') \
                       .replace('Target_L_R_binary', 'Target_L_R')
```


```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
from collections import Counter
import seaborn as sns

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define tract_display_mapping to map original tract names to unique simplified names
tract_display_mapping = {
    'ATR_L_fa': 'ATR L',
    'ATR_R_fa': 'ATR R',
    'CST_L_fa': 'CST L',
    'CST_R_fa': 'CST R',
    'CGC_L_fa': 'CGC L',
    'CGC_R_fa': 'CGC R',
    'CGH_L_fa': 'CGH L',
    'CGH_R_fa': 'CGH R',
    'FMA_fa': 'FMA',
    'FMI_fa': 'FMI',
    'IFOF_L_fa': 'IFOF L',
    'IFOF_R_fa': 'IFOF R',
    'ILF_L_fa': 'ILF L',
    'ILF_R_fa': 'ILF R',
    'SLF_L_fa': 'SLF L',
    'SLF_R_fa': 'SLF R',
    'SLFT_L_fa': 'SLFT L',
    'SLFT_R_fa': 'SLFT R',
    'UF_L_fa': 'UF L',
    'UF_R_fa': 'UF R',
    'PTR_L_fa': 'Thalamic Radiation L',
    'PTR_R_fa': 'Thalamic Radiation R'
}

#%% Define Modified Function
def run_lasso_fixed_alpha(target_variable, pivot_df, alpha, n_iterations, test_size, feature_names, tract_features, tract_display_mapping, selection_threshold=0.75, top_n=None):
    """
    Runs the LASSO regression process for a given target variable, using fixed alpha.
    Features are selected based on the highest absolute coefficient values after applying the selection threshold.
    
    Parameters:
    - target_variable: str, name of the target column in pivot_df
    - pivot_df: pd.DataFrame, the pivoted dataframe with features and target
    - alpha: float, regularization strength for LASSO
    - n_iterations: int, number of bootstrap iterations
    - test_size: float, proportion of the dataset to include in the test split
    - feature_names: list of str, feature column names
    - tract_features: list of str, names of tract features
    - tract_display_mapping: dict, mapping of tract feature names to simplified names
    - selection_threshold: float, proportion threshold for feature selection (e.g., 0.75 for 75%)
    - top_n: int or None, number of top features to select based on absolute coefficients. If None, select all.
    
    Returns:
    - most_common_feature_names: list of str, selected feature names
    - selected_mean_coefficients: list of float, mean coefficients of selected features
    - df_coefficients: pd.DataFrame, dataframe of selected features and their coefficients
    - df_tracts_top: pd.DataFrame, dataframe of selected tract features for plotting
    """
    print(f"\n{'='*80}")
    print(f"Processing Target Variable: {target_variable} with fixed alpha {alpha}")
    print(f"{'='*80}\n")
    
    # Prepare Target Variable
    y = pivot_df[target_variable].astype(float).values.ravel()
    X = pivot_df[feature_names].values  # Use the reshaped features
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    selected_features_list = []
    adjusted_r2_scores = []
    coefficients_list = []
    unconverged_count = 0
    
    for i in range(n_iterations):
        # Randomly split the data into training and testing sets with different seeds for variability
        X_resampled, y_resampled = resample(X, y, random_state=RANDOM_SEED + i)
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=test_size, random_state=RANDOM_SEED+i
        )
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Lasso Regression
        lasso = Lasso(alpha=alpha, max_iter=10000, random_state=RANDOM_SEED)
        lasso.fit(X_train_scaled, y_train)
        
        if lasso.n_iter_ == lasso.max_iter:
            unconverged_count +=1
        
        # Get indices of non-zero coefficients
        non_zero_indices = np.where(lasso.coef_ != 0)[0]
        selected_features_list.append(non_zero_indices)
        coefficients_list.append(lasso.coef_)
        
        if len(non_zero_indices) == 0:
            adjusted_r2_scores.append(np.nan)
            continue
        
        # Prepare features for linear regression
        X_train_selected = X_train_scaled[:, non_zero_indices]
        X_test_selected = X_test_scaled[:, non_zero_indices]
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_selected, y_train)  # fit on training data
        y_pred = lr_model.predict(X_test_selected)  # evaluate on test data
        
        # Calculate Adjusted R-squared
        r2 = r2_score(y_test, y_pred)
        n = X_test_selected.shape[0]
        p = X_test_selected.shape[1]
        if n - p - 1 == 0:
            adjusted_r2 = np.nan
        else:
            adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
        adjusted_r2_scores.append(adjusted_r2)
    
    # Calculate mean adjusted R-squared
    mean_adjusted_r2 = np.nanmean(adjusted_r2_scores)
    print(f"Mean Adjusted R-squared over {n_iterations} iterations: {mean_adjusted_r2:.4f}")
    print(f"Number of unconverged runs: {unconverged_count}")
    
    # Extract the most frequently selected features
    feature_counter = Counter()
    for features in selected_features_list:
        feature_counter.update(features)
    # Get features selected in more than the specified threshold
    threshold_count = n_iterations * selection_threshold
    most_common_features = [feat for feat, count in feature_counter.items() if count >= threshold_count]
    
    if not most_common_features:
        print("No features were selected based on the specified threshold.")
        return ([], [], pd.DataFrame(), pd.DataFrame())
    
    # Get the names of the most common features
    most_common_feature_names = [feature_names[i] for i in most_common_features]
    
    print(f"\nNumber of features meeting the {int(selection_threshold*100)}% threshold: {len(most_common_feature_names)}")
    print("Selected Features:")
    for name in most_common_feature_names:
        print(name)
    
    # Compute average coefficients for these features
    coefficients_array = np.array(coefficients_list)
    mean_coefficients = np.nanmean(coefficients_array, axis=0)
    selected_mean_coefficients = mean_coefficients[most_common_features]
    
    # Prepare DataFrame for plotting
    df_coefficients = pd.DataFrame({
        'Feature': most_common_feature_names,
        'Coefficient': selected_mean_coefficients
    })
    
    # Add absolute coefficients for sorting
    df_coefficients['Abs_Coefficient'] = df_coefficients['Coefficient'].abs()
    
    # Sort by absolute coefficient in descending order
    df_coefficients.sort_values(by='Abs_Coefficient', ascending=False, inplace=True)
    
    # Simplify feature names for better readability using tract_display_mapping
    def simplify_feature_name(name):
        if name in tract_features:
            return tract_display_mapping.get(name, name.replace('_fa', '').replace('_', ' '))
        else:
            name = name.replace('Sex_binary', 'Sex').replace('No_Leads_numeric', 'No Leads').replace('Target_L_R_binary', 'Target')
            name = name.replace('MDSUPDRSIIIpre_Percent_TOTAL_V', 'MDSUPDRS III')
            name = name.replace('Age_DBS_On', 'Age')
            return name
    
    df_coefficients['Feature_Simplified'] = df_coefficients['Feature'].apply(simplify_feature_name)
    
    # Separate tract features and clinical predictors
    df_tracts = df_coefficients[df_coefficients['Feature'].isin(tract_features)].copy()
    df_clinical = df_coefficients[~df_coefficients['Feature'].isin(tract_features)].copy()
    
    # Sort tracts by absolute coefficient magnitude
    df_tracts.sort_values(by='Abs_Coefficient', ascending=False, inplace=True)
    df_clinical.sort_values(by='Abs_Coefficient', ascending=False, inplace=True)
    
    # Optionally limit to top_n tracts based on absolute coefficients
    if top_n is not None:
        df_tracts_top = df_tracts.head(top_n)
    else:
        df_tracts_top = df_tracts
    
    # Plotting Tract Features Only
    plt.figure(figsize=(14, 10))
    sns.barplot(x='Feature_Simplified', y='Coefficient', data=df_tracts_top, palette='viridis')
    
    plt.title(f"Selected Tract Features for {target_variable.replace('_', ' ').capitalize()}", fontsize=24, weight='bold', color='black', pad=20)
    plt.ylabel("Average Coefficient", fontsize=22, weight='bold', color='black')
    plt.xlabel("", fontsize=22, weight='bold', color='black')
    plt.xticks(rotation=90, ha='center', fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"Selected_Tract_Features_{target_variable}.png", dpi=300)
    plt.show()
    
    # Plotting Tract Features and Clinical Predictors Together
    df_combined_top = pd.concat([df_tracts_top, df_clinical])
    
    plt.figure(figsize=(14, 10))
    sns.barplot(x='Feature_Simplified', y='Coefficient', data=df_combined_top, palette='viridis')
    
    plt.title(f"Selected Features for {target_variable.replace('_', ' ').capitalize()} (Including Clinical Predictors)", fontsize=24, weight='bold', color='black', pad=20)
    plt.ylabel("Average Coefficient", fontsize=22, weight='bold', color='black')
    plt.xlabel("", fontsize=22, weight='bold', color='black')
    plt.xticks(rotation=90, ha='center', fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"Selected_Features_With_Clinical_{target_variable}.png", dpi=300)
    plt.show()
    
    # Return values including df_tracts_top
    return (most_common_feature_names, selected_mean_coefficients,
            df_coefficients, df_tracts_top)

#%% Prepare Data (Reshaped for Tract-Specific FA Values)

# Replace the following line with your actual DataFrame loading method if necessary
# For example:
# relevant_tracts_csv = pd.read_csv('your_data.csv')

# Placeholder for relevant_tracts_csv. Replace this with your actual data loading.
# Example:
# relevant_tracts_csv = pd.read_csv('relevant_tracts.csv')

# Assuming relevant_tracts_csv is already loaded as a DataFrame
if 'relevant_tracts_csv' not in globals():
    raise ValueError("Please load your data into the 'relevant_tracts_csv' DataFrame before running this code.")

# Pivot the data to get tract-specific FA values as columns
pivot_df = relevant_tracts_csv.pivot_table(
    index=['subject_id', 'MDSUPDRSIIIpre_Percent_TOTAL_V',
           'delta_ledd1', 'delta_ledd2', 'delta_ledd3',
           'Target_L_R', 'Age_DBS_On', 'Sex', 'No_Leads'],
    columns='tractID', values='dti_fa'
).reset_index()

# Fill missing values only in numeric columns with the median values
pivot_df.fillna(pivot_df.select_dtypes(include=[np.number]).median(), inplace=True)

# Process 'Target_L_R', 'Sex', and 'No_Leads' features
pivot_df['Target_L_R_binary'] = process_target_lr(pivot_df['Target_L_R']).astype(int)
pivot_df['Sex_binary'] = process_sex(pivot_df['Sex']).astype(int)
pivot_df['No_Leads_numeric'] = process_no_leads(pivot_df['No_Leads']).astype(int)

# List of columns that are not tract features
non_tract_cols = [
    'subject_id', 'MDSUPDRSIIIpre_Percent_TOTAL_V',
    'delta_ledd1', 'delta_ledd2', 'delta_ledd3',
    'Target_L_R', 'Age_DBS_On', 'Sex', 'No_Leads',
    'Target_L_R_binary', 'Sex_binary', 'No_Leads_numeric'
]

# Define target variables and features
target_variables = ['delta_ledd1', 'delta_ledd2', 'delta_ledd3']

tract_features = [col for col in pivot_df.columns if col not in non_tract_cols]

clinical_features = ['MDSUPDRSIIIpre_Percent_TOTAL_V', 'Target_L_R_binary',
                     'Age_DBS_On', 'Sex_binary', 'No_Leads_numeric']

feature_columns = tract_features + clinical_features
X = pivot_df[feature_columns].values
feature_names = feature_columns

#%% Run Modified LASSO for Each Target Variable

alpha_fixed = 1  # Increased alpha for more stringent selection
n_iterations = 1000
test_size = 0.2
selection_threshold = 0.75  # 75% selection frequency
top_n = None  # Set to None to include all features meeting the threshold

# Initialize dictionaries to store results
top_features_ledd = {}
top_coefficients_ledd = {}
df_coefficients_ledd = {}

# Initialize lists to store tract names and coefficients for each target
top_tract_names_delta_ledd1 = []
top_tract_coefficients_delta_ledd1 = []
top_tract_names_delta_ledd2 = []
top_tract_coefficients_delta_ledd2 = []
top_tract_names_delta_ledd3 = []
top_tract_coefficients_delta_ledd3 = []

for target in target_variables:
    (top_features, top_coefficients,
     df_coefficients, df_tracts_top) = run_lasso_fixed_alpha(
        target_variable=target,
        pivot_df=pivot_df,
        alpha=alpha_fixed,
        n_iterations=n_iterations,
        test_size=test_size,
        feature_names=feature_names,
        tract_features=tract_features,
        tract_display_mapping=tract_display_mapping,
        selection_threshold=selection_threshold,
        top_n=top_n
    )
    # Store top features and coefficients
    top_features_ledd[target] = top_features
    top_coefficients_ledd[target] = top_coefficients
    df_coefficients_ledd[target] = df_coefficients

    # Assign tract names and coefficients from df_tracts_top
    if target == 'delta_ledd1':
        top_tract_names_delta_ledd1 = df_tracts_top['Feature'].tolist()
        top_tract_coefficients_delta_ledd1 = df_tracts_top['Coefficient'].tolist()
    elif target == 'delta_ledd2':
        top_tract_names_delta_ledd2 = df_tracts_top['Feature'].tolist()
        top_tract_coefficients_delta_ledd2 = df_tracts_top['Coefficient'].tolist()
    elif target == 'delta_ledd3':
        top_tract_names_delta_ledd3 = df_tracts_top['Feature'].tolist()
        top_tract_coefficients_delta_ledd3 = df_tracts_top['Coefficient'].tolist()

    # Create table for tract features
    df_top_tracts = pd.DataFrame({
        'Tract': df_tracts_top['Feature_Simplified'],
        'Coefficient': df_tracts_top['Coefficient']
    })
    print(f"\nSelected Tracts for {target}:")
    print(df_top_tracts.to_string(index=False))

#%% Combine Selected Tracts and Coefficients into a Single Table

df_all_tracts = pd.DataFrame()
for target in target_variables:
    df_temp = df_coefficients_ledd[target][['Feature_Simplified', 'Coefficient']].copy()
    df_temp.rename(columns={'Feature_Simplified': 'Feature'}, inplace=True)
    df_temp.rename(columns={'Coefficient': target}, inplace=True)
    if df_all_tracts.empty:
        df_all_tracts = df_temp
    else:
        df_all_tracts = pd.merge(df_all_tracts, df_temp, on='Feature', how='outer')

# Replace NaN with 0
df_all_tracts.fillna(0, inplace=True)

# Print the combined table
print("\nCombined Selected Tracts and Coefficients:")
print(df_all_tracts.to_string(index=False))

#%% Plot Combined Results

# Set the 'Feature' column as index for plotting
df_all_tracts.set_index('Feature', inplace=True)
df_all_tracts.plot(kind='bar', figsize=(14,10), colormap='viridis')

plt.ylabel('LASSO Coefficient', fontsize=22)
plt.title('Selected Tract Coefficients for Each LEDD Timepoint', fontsize=22)
plt.xticks(rotation=45, ha='right', fontsize=22)
plt.yticks(fontsize=20)
plt.legend(['0-3 mo.', '3-6 mo.', '6-12 mo.'], title='LEDD Timepoint', fontsize=15, title_fontsize=16)

# Save the plot
plt.tight_layout()
plt.savefig("Selected_Tract_Coefficients.png", dpi=300)

plt.show()

```

# Final Figure Outputs

# Cumulative Plots with Averaged Coefficients


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from IPython.display import display
import os
import dataframe_image as dfi  # Ensure this is correctly installed

def analyze_and_generate_publication_ready_outputs(
    data, 
    mdsupdrs_vars, 
    y_vars, 
    basic_covariates,
    wb_fa_covariates,
    group_labels, 
    tract_display_mapping,
    top_tracts_dict,
    top_coefficients_dict,
    output_dir='output',
    image_formats=['png']
):
    """
    Analyzes the data by running OLS regressions for different models and generates publication-ready
    tables and bar charts for R², Adjusted R², and AIC values. Exports each table and figure as a file.

    Adjustments based on user requirements:
    - Maintains blue and purple color scheme.
    - Sets tick labels to font size 22.
    - Sets axis titles to a larger font size (26).
    - Correctly overlays sample size labels only on the "Clinical only" bars.
    - Outputs the legend separately.
    - Generates summary tables for each subscore including R², Adjusted R², and AIC.
    - Bar charts are grouped with separate x positions for each model, and bars for each time point within each model.
    """

    # Make a copy of the data to avoid modifying the original DataFrame
    data = data.copy(deep=True)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure 'No_Leads' is treated as a categorical variable
    if 'No_Leads' in data.columns:
        data['No_Leads'] = data['No_Leads'].astype('category')

    # Abbreviate MDSUPDRS variables for labeling
    mdsupdrs_abbrev = {
        'MDSUPDRSIIIpre_Percent_TOTAL_V': 'Total',
        'MDSUPDRSIIIpre_Percent_BRADY': 'Brady',
        'MDSUPDRSIIIpre_Percent_TREMOR': 'Tremor',
        'MDSUPDRSIIIpre_Percent_AXIAL': 'Axial'
    }

    # Update dependent variable labels
    dep_var_labels = {
        'delta_ledd1': '0-3 mo.',
        'delta_ledd2': '3-6 mo.',
        'delta_ledd3': '6-12 mo.'
    }

    # Prepare a list to collect all results
    all_results = []

    # Verify that all covariate columns exist in the data
    required_columns = set()
    for y_var in y_vars:
        required_columns.update(top_tracts_dict[y_var])
    required_columns.update(basic_covariates)
    required_columns.update(wb_fa_covariates)
    required_columns.update(mdsupdrs_vars)
    required_columns.update(y_vars)

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        print(f"Warning: The following required columns are missing in the provided data: {missing_columns}")

    # Iterate over each MDSUPDRS variable
    for mdsupdrs_var in mdsupdrs_vars:
        # Initialize lists to store results for this subscore
        subscore_results_summary = []
        subscore_results_model_stats = []

        # Iterate over each model
        for idx, model_type in enumerate(group_labels):
            # For each model, iterate over each y_var (delta_ledd)
            for y_var in y_vars:
                # Get the covariates for the current model and y_var
                if model_type == 'Clinical only':
                    covariates = basic_covariates
                elif model_type == 'Clinical + wb-FA':
                    covariates = wb_fa_covariates
                elif model_type == 'Clinical + tractFA':
                    top_tracts = top_tracts_dict[y_var]
                    covariates = basic_covariates + top_tracts
                else:
                    covariates = basic_covariates  # Default to basic_covariates if unknown model_type

                # Prepare the formula
                formula = f"{y_var} ~ {mdsupdrs_var} + {' + '.join(covariates)}"

                # Drop missing values
                variables = [y_var, mdsupdrs_var] + covariates
                if not set(variables).issubset(data.columns):
                    print(f"Skipping model {model_type} for {dep_var_labels.get(y_var, y_var)} due to missing columns.")
                    continue
                model_data = data.dropna(subset=variables)

                # Check if enough data points
                if len(model_data) >= 2:
                    # Fit the model
                    reg_model = ols(formula, data=model_data).fit()

                    # Extract statistics
                    r_squared = reg_model.rsquared
                    adj_r_squared = reg_model.rsquared_adj
                    aic = reg_model.aic
                    n_obs = int(reg_model.nobs)
                    df_model = int(reg_model.df_model)
                    df_resid = int(reg_model.df_resid)

                    # Append results
                    subscore_results_summary.append({
                        'Model': model_type,
                        'Time Point': dep_var_labels.get(y_var, y_var),
                        'R²': round(r_squared, 3),
                        'Adjusted R²': round(adj_r_squared, 3),
                        'AIC': round(aic, 2)
                    })

                    subscore_results_model_stats.append({
                        'Model': model_type,
                        'Time Point': dep_var_labels.get(y_var, y_var),
                        '# Observations': n_obs
                    })
                else:
                    print(f"Not enough data for {model_type} with {mdsupdrs_abbrev.get(mdsupdrs_var, mdsupdrs_var)} at {dep_var_labels.get(y_var, y_var)}")

        # Create DataFrames for summary and model statistics
        subscore_df_summary = pd.DataFrame(subscore_results_summary)
        subscore_df_model_stats = pd.DataFrame(subscore_results_model_stats)

        # Append to all_results
        all_results.append((
            mdsupdrs_abbrev.get(mdsupdrs_var, mdsupdrs_var),
            subscore_df_summary,
            subscore_df_model_stats
        ))

    # Now, generate bar plots for Adjusted R²
    for mdsupdrs_label, df_summary, df_model_stats in all_results:
        # Pivot the data to get models on x-axis and time points as groups
        df_pivot = df_summary.pivot(
            index='Model',
            columns='Time Point',
            values='Adjusted R²'
        ).reindex(group_labels)

        # Prepare the data for plotting
        df_pivot = df_pivot.reset_index()
        time_point_order = [dep_var_labels[y_var] for y_var in y_vars]

        # Colors for each Time Point (blue and purple shades)
        colors = ['#1f77b4', '#9467bd', '#c5b0d5']  # Adjust as needed to maintain blue and purple scheme

        # Set up the plot
        plt.figure(figsize=(18, 10))
        bar_width = 0.2
        x = np.arange(len(group_labels))  # the label locations

        # Plot bars for each time point
        for i, time_point in enumerate(time_point_order):
            plt.bar(
                x + (i - 1) * bar_width,
                df_pivot[time_point],
                width=bar_width,
                color=colors[i],
                edgecolor='black',
                label=time_point
            )

        # Add sample size labels on top of the "Clinical only" bars
        clinical_only_data = df_model_stats[df_model_stats['Model'] == 'Clinical only']
        for i, time_point in enumerate(time_point_order):
            n_obs_row = clinical_only_data[clinical_only_data['Time Point'] == time_point]
            if not n_obs_row.empty:
                n_obs = n_obs_row['# Observations'].values[0]
                height = df_pivot[df_pivot['Model'] == 'Clinical only'][time_point].values[0]
                plt.text(
                    x[0] + (i - 1) * bar_width,
                    height + 0.003,
                    f'n = {n_obs}',
                    ha='center',
                    va='bottom',
                    fontsize=18,
                    fontweight='bold'
                )

        # Adjust x-axis labels
        plt.xticks(x, group_labels, fontsize=26, fontweight='bold')

        # Remove the "Model" axis label
        plt.xlabel('', fontsize=26, fontweight='bold')  # Removed the label text

        # Set y-axis label with increased font size
        plt.ylabel(f'Adjusted R² ({mdsupdrs_label})', fontsize=26, fontweight='bold')

        # Adjust y-axis tick labels font size
        plt.yticks(fontsize=26, fontweight='bold')

        # Set legend
        plt.legend(title='Time Point', fontsize=16, title_fontsize=18)

        # Create a separate figure for the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        fig_legend = plt.figure(figsize=(2, 2))
        fig_legend.legend(
            handles, labels,
            loc='center',
            frameon=False,
            fontsize=20
        )
        # Save the legend separately
        legend_filename = os.path.join(output_dir, f"legend_{mdsupdrs_label}.png")
        try:
            fig_legend.savefig(legend_filename, dpi=300, bbox_inches='tight')
            print(f"Saved legend to {legend_filename}")
        except Exception as e:
            print(f"Error saving legend {legend_filename}: {e}")
        plt.close(fig_legend)  # Close the legend figure to free memory

        # Remove legend from main plot
        plt.legend([], [], frameon=False)

        # Remove the grid for a cleaner look
        sns.despine()

        # Adjust layout
        plt.tight_layout()

        # Save the figure in specified formats
        for fmt in image_formats:
            fig_filename = os.path.join(output_dir, f"Adjusted_R2_{mdsupdrs_label}_Model.{fmt}")
            try:
                plt.savefig(fig_filename, dpi=300, format=fmt, bbox_inches='tight')
                print(f"Saved bar chart to {fig_filename}")
            except Exception as e:
                print(f"Error saving {fig_filename}: {e}")

        plt.show()

        # Output summary statistics
        print(f"\n=== {mdsupdrs_label} Model ===\n")

        # Display Model Summary Table
        print("**Model Summary:**\n")
        styled_summary = df_summary.style.set_properties(**{
            'font-size': '16pt',
            'font-weight': 'bold'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('font-size', '16pt'), 
                ('font-weight', 'bold'), 
                ('background-color', '#f2f2f2')
            ]},
            {'selector': 'td', 'props': [('font-size', '16pt')]}
        ])

        display(styled_summary)

        # Export Model Summary Table to Excel
        summary_excel_filename = os.path.join(output_dir, f"{mdsupdrs_label}_model_summary.xlsx")
        try:
            df_summary.to_excel(summary_excel_filename, index=False)
            print(f"Saved model summary table to {summary_excel_filename}")
        except Exception as e:
            print(f"Error saving model summary table {summary_excel_filename}: {e}")

        # Export the styled table as an image using dataframe_image
        table_image_filename = os.path.join(output_dir, f"{mdsupdrs_label}_model_summary.png")
        try:
            dfi.export(styled_summary, table_image_filename)
            print(f"Saved model summary table image to {table_image_filename}")
        except Exception as e:
            print(f"Error saving model summary table image {table_image_filename}: {e}")



# Mapping of simplified tract names to actual column names (with your edits)
tract_name_mapping = {
    'ATR_L_fa': 'ATR L',
    'ATR_R_fa': 'ATR R',
    'CST_L_fa': 'CST L',
    'CST_R_fa': 'CST R',
    'CGC_L_fa': 'CGC L',
    'CGC_R_fa': 'CGC R',
    'CGH_L_fa': 'CGH L',
    'CGH_R_fa': 'CGH R',
    'FMA_fa': 'FMA',
    'FMI_fa': 'FMI',
    'IFOF_L_fa': 'IFOF L',
    'IFOF_R_fa': 'IFOF R',
    'ILF_L_fa': 'ILF L',
    'ILF_R_fa': 'ILF R',
    'SLF_L_fa': 'SLF L',
    'SLF_R_fa': 'SLF R',
    'SLFT_L_fa': 'SLFT L',
    'SLFT_R_fa': 'SLFT R',
    'UF_L_fa': 'UF L',
    'UF_R_fa': 'UF R',
    'PTR_L_fa': 'Thalamic Radiation L',
    'PTR_R_fa': 'Thalamic Radiation R'
}



# Map simplified tract names to actual column names for each y_var
top_tracts_dict = {
    'delta_ledd1': top_tract_names_delta_ledd1,
    'delta_ledd2': top_tract_names_delta_ledd2,
    'delta_ledd3': top_tract_names_delta_ledd3,
}


# Define your dependent variables
y_vars = ['delta_ledd1', 'delta_ledd2', 'delta_ledd3']

# Define your covariates for each model
basic_covariates = ['Target_L_R', 'Age_DBS_On', 'No_Leads', 'Sex']
wb_fa_covariates = basic_covariates + ['FA_15']

# Define group_labels
group_labels = [
    'Clinical only', 
    'Clinical + wb-FA', 
    'Clinical + tractFA'
]
top_coefficients_dict = {
    'delta_ledd1': top_tract_coefficients_delta_ledd1,
    'delta_ledd2': top_tract_coefficients_delta_ledd2,
    'delta_ledd3': top_tract_coefficients_delta_ledd3,
}
# Define mdsupdrs_vars
mdsupdrs_vars = [
    'MDSUPDRSIIIpre_Percent_TOTAL_V',
    'MDSUPDRSIIIpre_Percent_BRADY',
    'MDSUPDRSIIIpre_Percent_TREMOR',
    'MDSUPDRSIIIpre_Percent_AXIAL'
]

# Now, run the analysis
analyze_and_generate_publication_ready_outputs(
    data=patient_df,  # Replace with your actual DataFrame
    mdsupdrs_vars=mdsupdrs_vars, 
    y_vars=y_vars, 
    basic_covariates=basic_covariates,
    wb_fa_covariates=wb_fa_covariates,
    group_labels=group_labels,
    tract_display_mapping=tract_display_mapping,
    top_tracts_dict=top_tracts_dict,
    top_coefficients_dict=top_coefficients_dict,
    output_dir='output',
    image_formats=['png', 'jpg', 'pdf']
)


```


```python
# Import necessary libraries
import nibabel as nib
from nilearn import image, plotting, datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

# Set Helvetica as the default font
rcParams['font.family'] = 'Helvetica'

# Define the mapping from tract names to intensity values based on the atlas
tract_mapping = {
    'ATR L': 1,    # Anterior thalamic radiation (left)
    'ATR R': 2,    # Anterior thalamic radiation (right)
    'CST L': 3,    # Corticospinal tract (left)
    'CST R': 4,    # Corticospinal tract (right)
    'CGC L': 5,    # Cingulum (cingulate gyrus) left
    'CGC R': 6,    # Cingulum (cingulate gyrus) right
    'CGH L': 7,    # Cingulum (hippocampus) left
    'CGH R': 8,    # Cingulum (hippocampus) right
    'FMA': 9,      # Forceps major
    'FMI': 10,     # Forceps minor
    'IFOF L': 11,  # Inferior fronto-occipital fasciculus left
    'IFOF R': 12,  # Inferior fronto-occipital fasciculus right
    'ILF L': 13,   # Inferior longitudinal fasciculus left
    'ILF R': 14,   # Inferior longitudinal fasciculus right
    'SLF L': 15,   # Superior longitudinal fasciculus left
    'SLF R': 16,   # Superior longitudinal fasciculus right
    'SLFT L': 17,  # Superior longitudinal fasciculus (temporal part) left
    'SLFT R': 18,  # Superior longitudinal fasciculus (temporal part) right
    'UF L': 19,    # Uncinate fasciculus left
    'UF R': 20     # Uncinate fasciculus right
}

# Load the manually downloaded atlas using relative path
atlas_dir = os.path.join('..', 'data', 'atlas')
atlas_path = os.path.join(atlas_dir, 'JHU-ICBM-tracts-maxprob-thr25-2mm.nii.gz')
atlas_img = nib.load(atlas_path)

# Load the MNI template
template = datasets.load_mni152_template()

# Define cut coordinates for plotting
cut_coords_sagittal = [-20, 0, 20]  # Three slices for sagittal view
cut_coords_coronal = [-30, 0, 30]   # Three slices for coronal view
cut_coords_axial = [-20, 0, 20]     # Three slices for axial view
cut_coords = [cut_coords_sagittal, cut_coords_coronal, cut_coords_axial]
display_modes = ['x', 'y', 'z']  # sagittal, coronal, axial views

# Function to simplify tract names
def simplify_tract_name(tract_name):
    """
    Simplify the tract name by removing prefixes and suffixes.
    """
    tract = tract_name.replace('_fa', '').replace('_', ' ').strip()
    return tract

# Function to plot top tracts
def plot_top_tracts(top_tract_names, top_tract_coefficients, delta_id):
    """
    Plot the top tracts based on LASSO coefficients.

    Parameters:
    - top_tract_names: List of tract names (e.g., ['CST L', 'ILF R', ...])
    - top_tract_coefficients: List of coefficients corresponding to the tracts
    - delta_id: Identifier for delta_ledd (e.g., 'delta_ledd1')
    """
    # Simplify tract names
    top_tracts_simplified = [simplify_tract_name(name) for name in top_tract_names]

    # Verify that all simplified tract names are present in the mapping
    missing_tracts = [tract for tract in top_tracts_simplified if tract not in tract_mapping]
    if missing_tracts:
        print(f"Warning ({delta_id}): The following tracts are missing in the tract_mapping and will be skipped: {missing_tracts}")

    # Filter out any tracts not present in the mapping
    valid_indices = [i for i, tract in enumerate(top_tracts_simplified) if tract in tract_mapping]
    top_tracts_simplified = [top_tracts_simplified[i] for i in valid_indices]
    top_coefficients_filtered = [top_tract_coefficients[i] for i in valid_indices]

    # Select only the top 10 tracts (if not already)
    top_tracts_simplified = top_tracts_simplified[:10]
    top_coefficients_filtered = top_coefficients_filtered[:10]

    # Create a list of intensity values corresponding to the top tracts
    top_intensities = [tract_mapping[tract] for tract in top_tracts_simplified]

    # Normalize the Lasso coefficients for colormap mapping
    vmin, vmax = min(top_coefficients_filtered), max(top_coefficients_filtered)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Use the reversed rainbow colormap for better visual distinction
    cmap = cm.get_cmap('rainbow')

    # Generate colors for each tract based on normalized coefficients
    colors = cmap(norm(top_coefficients_filtered))

    # Create a mapping from tract name to its color
    tract_to_color = {tract: colors[i] for i, tract in enumerate(top_tracts_simplified)}

    # Create a custom colormap with discrete colors for each tract
    custom_cmap = ListedColormap([tract_to_color[tract] for tract in top_tracts_simplified])

    # Create an empty array to hold the combined tracts with discrete labels
    combined_tract_data = np.zeros(atlas_img.shape, dtype=np.int16)

    # Assign unique integer labels to each tract
    tract_label_mapping = {tract: i+1 for i, tract in enumerate(top_tracts_simplified)}  # Labels start at 1

    for tract in top_tracts_simplified:
        intensity = tract_mapping[tract]
        label = tract_label_mapping[tract]
        tract_data = (atlas_img.get_fdata() == intensity)
        combined_tract_data[tract_data] = label

    # Create a Nifti image for the combined tracts
    combined_tract_img = nib.Nifti1Image(combined_tract_data, atlas_img.affine)

    # Resample the combined image to the MNI template space with nearest interpolation
    resampled_tract_img = image.resample_to_img(combined_tract_img, template, interpolation='nearest')

    # Set up the figure for 3x3 plot, stacking views vertically
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), facecolor='white')  # Increased figure size for clarity

    # Loop through the views and plot 3 slices for each view
    for i, display_mode in enumerate(display_modes):
        for j, cut_coord in enumerate(cut_coords[i]):
            ax = axes[i, j]
            display = plotting.plot_roi(
                resampled_tract_img,
                bg_img=template,
                display_mode=display_mode,
                cut_coords=[cut_coord],
                axes=ax,
                cmap=custom_cmap,
                colorbar=False,  # Disable individual color bars
                threshold=0.5,   # Since labels start at 1
                black_bg=False,
                annotate=False,   # Turn off annotations including L/R labels
            )
            # Remove axis labels (x, y, z)
            ax.set_xticks([])
            ax.set_yticks([])

    # Remove spaces between subplots (minimal hspace and wspace)
    plt.subplots_adjust(wspace=0, hspace=0.001)

    # Add a border around the entire figure
    for ax in axes.flatten():
        for spine in ax.spines.values():
            spine.set_visible(True)

    # Define output paths dynamically based on delta_id, saving to the results/figures directory
    figure_output_dir = os.path.join('..', 'results', 'figures')
    os.makedirs(figure_output_dir, exist_ok=True) # Ensure the directory exists
    output_image_path = os.path.join(figure_output_dir, f'{delta_id}_tract_stacked_visualization_bordered.png')
    colorbar_image_path = os.path.join(figure_output_dir, f'{delta_id}_colorbar_vertical_rainbow_inverted.png')

    # Save the plot as an image file (png) with a white background
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=300, facecolor='white')
    plt.close(fig)  # Close the figure to free memory

    # Create a separate figure for the color bar, making it vertical
    fig_colorbar, ax_colorbar = plt.subplots(figsize=(2, 8))  # Taller and narrow for vertical color bar

    # Create a list of colors corresponding to each tract
    color_list = [tract_to_color[tract] for tract in top_tracts_simplified]

    # Create a ListedColormap for the colorbar
    colorbar_cmap = ListedColormap(color_list)

    # Create boundaries and norm for discrete colorbar
    bounds = np.arange(len(top_tracts_simplified) + 1)
    norm_cb = BoundaryNorm(bounds, colorbar_cmap.N)

    # Create the colorbar
    cbar = plt.colorbar(
        cm.ScalarMappable(norm=norm_cb, cmap=colorbar_cmap),
        cax=ax_colorbar,
        boundaries=bounds,
        ticks=np.arange(len(top_tracts_simplified)) + 0.5,
        spacing='uniform',
        orientation='vertical'
    )

    # Set the labels for each color in the colorbar
    tract_labels = top_tracts_simplified  # Already simplified

    # Adjust tract labels for display (optional: replace underscores with spaces)
    tract_labels_display = [label.replace('_', ' ') for label in tract_labels]

    cbar.ax.set_yticklabels(tract_labels_display, fontsize=12)

    # Set colorbar label
    cbar.set_label('Tracts Colored by LASSO Coefficient', fontsize=14)

    # Adjust layout for the colorbar
    plt.tight_layout()

    # Save the vertical colorbar as a separate image
    plt.savefig(colorbar_image_path, bbox_inches='tight', pad_inches=0, dpi=300, facecolor='white')
    plt.close(fig_colorbar)  # Close the colorbar figure to free memory

    print(f"Plots saved for {delta_id}:")
    print(f" - Tract Visualization: {output_image_path}")
    print(f" - Colorbar: {colorbar_image_path}\n")

# Example usage:
# Assuming the following variables are defined:
# top_tract_names_delta_ledd1, top_tract_coefficients_delta_ledd1
# top_tract_names_delta_ledd2, top_tract_coefficients_delta_ledd2
# top_tract_names_delta_ledd3, top_tract_coefficients_delta_ledd3

# Replace the following lines with your actual data
# For demonstration purposes, here's some dummy data:
# Remove or replace these dummy definitions with your actual data variables.

# Dummy data for demonstration (remove in actual usage)
# top_tract_names_delta_ledd1 = ['CST L', 'ILF R', 'ATR L']
# top_tract_coefficients_delta_ledd1 = [0.5, -0.3, 0.2]
# top_tract_names_delta_ledd2 = ['CGC L', 'FMA', 'IFOF R']
# top_tract_coefficients_delta_ledd2 = [0.6, -0.2, 0.4]
# top_tract_names_delta_ledd3 = ['SLF L', 'UF R', 'CGH L']
# top_tract_coefficients_delta_ledd3 = [0.7, -0.1, 0.3]

# Uncomment and use your actual data variables below:
# Plot for delta_ledd1
plot_top_tracts(top_tract_names_delta_ledd1, top_tract_coefficients_delta_ledd1, 'delta_ledd1')

# Plot for delta_ledd2
plot_top_tracts(top_tract_names_delta_ledd2, top_tract_coefficients_delta_ledd2, 'delta_ledd2')

# Plot for delta_ledd3
plot_top_tracts(top_tract_names_delta_ledd3, top_tract_coefficients_delta_ledd3, 'delta_ledd3')

# Display a final message
print("All plots have been generated and saved successfully.")


```
