# Tract-Specific FAQ Analyses
# Philip Shih

# Section I: Setup and Data Preparation
### Import libraries.


```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_ind
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
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_curve, roc_auc_score
from sklearn.utils import resample
from sklearn.exceptions import ConvergenceWarning
import random
import warnings
from collections import Counter, defaultdict
import nibabel as nib
from nilearn import image, plotting, datasets
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, BoundaryNorm
from IPython.display import display
import dataframe_image as dfi # Optional: for exporting styled tables as images

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Plotting defaults
rcParams['font.family'] = 'Helvetica'
sns.set(style="white", rc={'axes.grid': False, 'figure.dpi': 150}) 
```

### Define File Paths and Parameters

```python
# --- Configuration ---
DATA_DIR = os.path.join('..', 'data')
RESULTS_DIR = os.path.join('..', 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
ATLAS_DIR = os.path.join(DATA_DIR, 'atlas')

MRTRIX_CSV_DIR = os.path.join(RAW_DATA_DIR, 'mrtrix_csvs')
OUTCOME_FILE = os.path.join(METADATA_DIR, 'Deidentified_master_spreadsheet_01.07.25_SD.xlsx')
WB_FA_FILE = os.path.join(METADATA_DIR, 'FA_erosion_results.csv')
ATLAS_FILE = os.path.join(ATLAS_DIR, 'JHU-ICBM-tracts-maxprob-thr25-2mm.nii.gz')

# Analysis Parameters
TEST_SPLIT_SIZE = 0.20
LASSO_ALPHA = 1.0 # Fixed alpha for LASSO feature selection
LASSO_ITERATIONS = 1000 # Number of bootstrap iterations for LASSO
LASSO_SELECTION_THRESHOLD = 0.75 # Feature must be selected in >= 75% of iterations

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Data Directory: {os.path.abspath(DATA_DIR)}")
print(f"Results Directory: {os.path.abspath(RESULTS_DIR)}")
print(f"Figures Directory: {os.path.abspath(FIGURES_DIR)}")
```

### Load and Combine Tract FA Data

```python
# Regex to match the filenames
pattern = r'sub-PDa\d{3}_ses-01_dwi_space-RASMM_model-probCSD_algo-reco80_desc-profiles_dwi.csv'

# List all files in MRTRIX_CSV_DIR and filter based on the pattern
try:
    afq_files = [os.path.join(MRTRIX_CSV_DIR, f) for f in os.listdir(MRTRIX_CSV_DIR) if re.match(pattern, f)]
except FileNotFoundError:
    print(f"Error: Directory not found - {MRTRIX_CSV_DIR}")
    afq_files = []

if not afq_files:
     print(f"Warning: No AFQ profile CSV files found matching the pattern in {MRTRIX_CSV_DIR}")
     combined_fa_data = pd.DataFrame(columns=['nodeID', 'dti_fa', 'tractID', 'subject_id']) # Create empty df
else:
    fa_data_list = []
    # Loop through the file list and read each file into a dataframe
    for file in afq_files:
        try:
            df = pd.read_csv(file)
            # Extract the subject ID from the file name using os.path.basename for cross-platform compatibility
            subject_id = os.path.basename(file).split('_')[0]
            # Add a column with the subject ID
            df['subject_id'] = subject_id
            # Append the dataframe to the list
            fa_data_list.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    # Combine all dataframes into one
    if fa_data_list:
        combined_fa_data = pd.concat(fa_data_list, ignore_index=True)
        print(f"Loaded data from {len(fa_data_list)} subject CSV files.")
    else:
        print("No valid FA data loaded.")
        combined_fa_data = pd.DataFrame(columns=['nodeID', 'dti_fa', 'tractID', 'subject_id'])

# Check the first few rows of the combined dataframe
print("Combined FA Data Head:")
display(combined_fa_data.head())
```

### Load and Prepare Clinical Outcome Data

```python
try:
    outcome_data_raw = pd.read_excel(OUTCOME_FILE)
    print(f"Loaded outcome data from: {OUTCOME_FILE}")
except FileNotFoundError:
    print(f"Error: Outcome file not found - {OUTCOME_FILE}")
    outcome_data_raw = pd.DataFrame() # Create empty df

if not outcome_data_raw.empty:
    # Rename columns for consistency
    outcome_data = outcome_data_raw.rename(columns={
        'PTID_Retro_Clin': 'subject_id', 
        'LEDDpost1_Delta': 'delta_ledd1', 
        'LEDDpost2_Delta': 'delta_ledd2', 
        'LEDDpost3_Delta': 'delta_ledd3'
        # Add other renames if necessary
    })

    # Ensure the subject_id format matches the one used in FA data ('sub-PDaXXX')
    if 'subject_id' in outcome_data.columns:
         # Handle potential non-string values before applying string methods
        outcome_data['subject_id'] = outcome_data['subject_id'].astype(str)
        outcome_data['subject_id'] = outcome_data['subject_id'].apply(lambda x: f"sub-{x}" if not x.startswith('sub-') else x)
        outcome_data = outcome_data.drop_duplicates(subset=['subject_id'])
        
        # Convert relevant columns to numeric, coercing errors
        for col in ['delta_ledd1', 'delta_ledd2', 'delta_ledd3']:
             if col in outcome_data.columns:
                  outcome_data[col] = pd.to_numeric(outcome_data[col], errors='coerce')
        
        # Clean string columns (like Target_L_R, Sex)
        for col in outcome_data.select_dtypes(include='object').columns:
             outcome_data[col] = outcome_data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        print("Outcome Data Head after cleaning:")
        display(outcome_data.head())
    else:
        print("Warning: 'PTID_Retro_Clin' column not found in outcome data.")
        outcome_data = pd.DataFrame()
else:
     outcome_data = pd.DataFrame()
```

### Merge FA Data and Clinical Data

```python
if not combined_fa_data.empty and not outcome_data.empty:
    # Merge the FA data with the outcome measure data on the subject_id
    data_merged = pd.merge(combined_fa_data, outcome_data, on='subject_id', how='inner')
    print("Merged FA and Outcome Data Head:")
    display(data_merged.head())
    print(f"Number of rows after merge: {len(data_merged)}")
else:
    print("Skipping merge due to empty FA or outcome data.")
    data_merged = pd.DataFrame() 
```

### Aggregate Tract FA Values

```python
if not data_merged.empty:
    # Group data by subject and tract to calculate average FA value for each
    average_fa_per_subject_tract = data_merged.groupby(['subject_id', 'tractID'])['dti_fa'].mean().reset_index()
    
    # Merge averaged FA values back with unique clinical data per subject
    # First get unique clinical data rows
    clinical_data_unique = data_merged.drop(columns=['nodeID', 'dti_fa', 'tractID']).drop_duplicates(subset=['subject_id'])
    
    tract_specific_data = pd.merge(average_fa_per_subject_tract, clinical_data_unique, on='subject_id', how='inner')
    
    print("Tract Specific Data Head:")
    display(tract_specific_data.head())
    print(f"Number of rows in tract_specific_data: {len(tract_specific_data)}")
    
    # Save intermediate file
    tract_specific_data.to_csv(os.path.join(RESULTS_DIR, 'tract_specific_data_intermediate.csv'), index=False)
else:
    print("Skipping tract aggregation due to empty merged data.")
    tract_specific_data = pd.DataFrame()
```

### Filter Tracts Based on Presence

```python
if not tract_specific_data.empty:
    total_subjects = tract_specific_data['subject_id'].nunique()
    min_subjects_threshold = 100 # As per original code

    # Group by 'tractID' and count unique 'subject_ID'
    tract_counts = tract_specific_data.groupby('tractID')['subject_id'].nunique()

    # Filter 'tractID' that are present for at least min_subjects_threshold subjects
    valid_tracts = tract_counts[tract_counts >= min_subjects_threshold].index
    
    # Filter the dataframe
    relevant_tract_data = tract_specific_data[tract_specific_data['tractID'].isin(valid_tracts)].copy() # Use .copy()

    print(f"Filtered tracts present in >= {min_subjects_threshold} subjects. Kept {len(valid_tracts)} tracts.")
    print("Relevant Tract Data Head:")
    display(relevant_tract_data.head())
    
    # Save intermediate file
    relevant_tract_data.to_csv(os.path.join(RESULTS_DIR, 'relevant_tract_data_intermediate.csv'), index=False)
else:
    print("Skipping tract filtering due to empty tract_specific_data.")
    relevant_tract_data = pd.DataFrame()
    valid_tracts = [] # Ensure valid_tracts is defined
```

### Create Final Patient DataFrame (Pivot Tracts)

```python
if not relevant_tract_data.empty:
    # Pivot the relevant tract data
    pivot_df_tracts = relevant_tract_data.pivot_table(
        index='subject_id', 
        columns='tractID', 
        values='dti_fa'
    )
    
    # Get the unique clinical data again
    clinical_data_unique = relevant_tract_data.drop(columns=['tractID', 'dti_fa']).drop_duplicates(subset=['subject_id']).set_index('subject_id')
    
    # Join clinical data with pivoted tract data
    patient_df_prelim = clinical_data_unique.join(pivot_df_tracts)
    
    # Load and merge whole brain FA data
    try:
        wholebrain_df = pd.read_csv(WB_FA_FILE)
        if 'ID' in wholebrain_df.columns:
             # Ensure subject_id format matches
             wholebrain_df['subject_id'] = wholebrain_df['ID'].astype(str).str.replace('_fa.nii.gz', '', regex=False)
             wholebrain_df['subject_id'] = wholebrain_df['subject_id'].apply(lambda x: f"sub-{x}" if not x.startswith('sub-') else x)
             wholebrain_df = wholebrain_df.drop(columns=['ID'])
             
             # Merge with patient_df
             patient_df_merged = pd.merge(patient_df_prelim.reset_index(), wholebrain_df, on='subject_id', how='left').set_index('subject_id')
             print("Merged whole brain FA data.")
        else:
             print("Warning: 'ID' column not found in whole brain FA file. Skipping merge.")
             patient_df_merged = patient_df_prelim.copy()

    except FileNotFoundError:
        print(f"Warning: Whole brain FA file not found ({WB_FA_FILE}). Skipping merge.")
        patient_df_merged = patient_df_prelim.copy()
    except Exception as e:
        print(f"Error merging whole brain FA data: {e}")
        patient_df_merged = patient_df_prelim.copy()

    # --- Data Cleaning and Preprocessing for Modeling ---
    
    # Define key columns needed for analysis
    delta_ledd_targets = ['delta_ledd1', 'delta_ledd2', 'delta_ledd3']
    mdsupdrs_predictors = [
        'MDSUPDRSIIIpre_Percent_TOTAL_V',
        'MDSUPDRSIIIpre_Percent_BRADY',
        'MDSUPDRSIIIpre_Percent_TREMOR',
        'MDSUPDRSIIIpre_Percent_AXIAL'
    ]
    basic_clinical_covariates = ['Target_L_R', 'Age_DBS_On', 'No_Leads', 'Sex']
    wb_fa_predictor = ['FA_15'] # Assuming FA_15 is the whole-brain predictor
    
    # Ensure tract features are correctly identified (columns not in other lists)
    all_non_tract_cols = set(delta_ledd_targets + mdsupdrs_predictors + basic_clinical_covariates + wb_fa_predictor + ['subject_id', 'Unnamed: 0', 'Unnamed: 0.1']) # Add index/unnamed if they exist
    tract_feature_cols = [col for col in patient_df_merged.columns if col not in all_non_tract_cols and '_fa' in col] # Heuristic: tract cols contain '_fa'

    # Process categorical/binary clinical covariates
    patient_df_processed = patient_df_merged.copy()
    patient_df_processed['Target_L_R_binary'] = process_target_lr(patient_df_processed['Target_L_R'])
    patient_df_processed['Sex_binary'] = process_sex(patient_df_processed['Sex'])
    patient_df_processed['No_Leads_numeric'] = process_no_leads(patient_df_processed['No_Leads'])

    # Define final feature sets
    final_basic_covariates = ['Target_L_R_binary', 'Age_DBS_On', 'No_Leads_numeric', 'Sex_binary']
    final_wb_fa_covariates = final_basic_covariates + wb_fa_predictor
    
    # Drop rows with missing values in essential columns for modeling
    essential_cols = delta_ledd_targets + mdsupdrs_predictors + final_basic_covariates + wb_fa_predictor + tract_feature_cols
    patient_df_final = patient_df_processed.dropna(subset=essential_cols).copy()
    
    print(f"Final dataset size after dropping NaNs in essential columns: {patient_df_final.shape[0]} subjects.")

    # Save the final processed dataframe
    patient_df_final.reset_index().to_csv(os.path.join(RESULTS_DIR, 'patient_df_final_processed.csv'), index=False)
    
    print("Final Processed Patient Data Head:")
    display(patient_df_final.head())

else:
    print("Skipping final patient dataframe creation.")
    patient_df_final = pd.DataFrame()
    tract_feature_cols = []
    final_basic_covariates = []
    final_wb_fa_covariates = []
    mdsupdrs_predictors = []
    delta_ledd_targets = []

```

# Section II: Train/Test Split

```python
if not patient_df_final.empty:
    # Define features (X) and targets (Y) - we'll select specific targets later
    feature_cols = final_basic_covariates + final_wb_fa_covariates + tract_feature_cols
    # Remove duplicates just in case FA_15 was in basic_covariates
    feature_cols = sorted(list(set(feature_cols))) 
    
    X = patient_df_final[feature_cols]
    Y = patient_df_final[delta_ledd_targets + mdsupdrs_predictors] # Keep all potential targets/predictors together for now

    # Perform the 80/20 split
    # Stratify by 'Target_L_R_binary' if possible (check if column exists and has enough samples per class)
    stratify_col = None
    if 'Target_L_R_binary' in patient_df_final.columns and patient_df_final['Target_L_R_binary'].nunique() > 1:
         min_class_count = patient_df_final['Target_L_R_binary'].value_counts().min()
         # Need at least 2 samples per class for stratification with test_size=0.2
         if min_class_count >= 2:
              stratify_col = patient_df_final['Target_L_R_binary']
              print("Stratifying train/test split by 'Target_L_R_binary'.")
         else:
              print("Warning: Not enough samples in each 'Target_L_R_binary' class for stratification.")
    else:
         print("Stratification column 'Target_L_R_binary' not suitable or not found.")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_SEED,
        stratify=stratify_col
    )

    # Combine Y_train and Y_test back with X_train and X_test for easier handling in functions
    train_df = pd.concat([X_train, Y_train], axis=1)
    test_df = pd.concat([X_test, Y_test], axis=1)

    print(f"Training set size: {train_df.shape[0]}")
    print(f"Test set size: {test_df.shape[0]}")
else:
    print("Skipping train/test split due to empty data.")
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
```

# Section III: Feature Selection (LASSO)

```python
# --- LASSO Feature Selection Function ---
# (Modified to run on training data only)
def run_lasso_feature_selection(target_variable, train_data, feature_names, 
                                alpha, n_iterations, selection_threshold):
    """
    Runs LASSO with bootstrapping on training data to find stable features.
    """
    print(f"\n--- Running LASSO for target: {target_variable} ---")
    
    # Prepare Target Variable and Features from training data
    y_train = train_data[target_variable].astype(float).values.ravel()
    X_train_full = train_data[feature_names].values 
    
    n_samples = X_train_full.shape[0]
    
    if n_samples == 0:
        print("Warning: Training data is empty. Skipping LASSO.")
        return [], pd.DataFrame()

    selected_features_indices_list = []
    coefficients_list = []
    
    for i in range(n_iterations):
        # Bootstrap resampling of the training data
        X_resampled, y_resampled = resample(X_train_full, y_train, random_state=RANDOM_SEED + i)
        
        # Standardize features for this bootstrap sample
        scaler = StandardScaler()
        X_resampled_scaled = scaler.fit_transform(X_resampled)
        
        # Lasso Regression
        lasso = Lasso(alpha=alpha, max_iter=10000, random_state=RANDOM_SEED)
        lasso.fit(X_resampled_scaled, y_resampled)
        
        # Store coefficients and selected features
        coefficients_list.append(lasso.coef_)
        non_zero_indices = np.where(lasso.coef_ != 0)[0]
        selected_features_indices_list.append(non_zero_indices)
        
    # --- Feature Stability Analysis ---
    feature_counter = Counter()
    for indices in selected_features_indices_list:
        feature_counter.update(indices)
        
    threshold_count = n_iterations * selection_threshold
    stable_feature_indices = [idx for idx, count in feature_counter.items() if count >= threshold_count]
    
    if not stable_feature_indices:
        print(f"No features met the {selection_threshold*100}% stability threshold for {target_variable}.")
        return [], pd.DataFrame()

    stable_feature_names = [feature_names[i] for i in stable_feature_indices]
    
    # Compute average coefficients for stable features
    coefficients_array = np.array(coefficients_list)
    mean_coefficients = np.nanmean(coefficients_array, axis=0)
    stable_mean_coefficients = mean_coefficients[stable_feature_indices]

    df_stable_coeffs = pd.DataFrame({
        'Feature': stable_feature_names,
        'Mean_Coefficient': stable_mean_coefficients
    })
    df_stable_coeffs['Abs_Coefficient'] = df_stable_coeffs['Mean_Coefficient'].abs()
    df_stable_coeffs.sort_values(by='Abs_Coefficient', ascending=False, inplace=True)

    print(f"Identified {len(stable_feature_names)} stable features for {target_variable} (threshold={selection_threshold*100}%).")
    
    return stable_feature_names, df_stable_coeffs

# --- Run LASSO for each delta_ledd target ---
stable_features_per_ledd = {}
stable_coeffs_dfs = {}

if not train_df.empty:
    # Define all potential features for LASSO
    lasso_feature_pool = final_basic_covariates + tract_feature_cols # Exclude WB FA for tract selection? Or include? Let's include for now.
    lasso_feature_pool = sorted(list(set(lasso_feature_pool)))

    for target in delta_ledd_targets:
        stable_names, stable_df = run_lasso_feature_selection(
            target_variable=target,
            train_data=train_df,
            feature_names=lasso_feature_pool,
            alpha=LASSO_ALPHA,
            n_iterations=LASSO_ITERATIONS,
            selection_threshold=LASSO_SELECTION_THRESHOLD
        )
        stable_features_per_ledd[target] = stable_names
        stable_coeffs_dfs[target] = stable_df
        
        # Optional: Plot stable coefficients for this target
        if not stable_df.empty:
             plt.figure(figsize=(12, 6))
             # Simplify names for plotting
             stable_df['Feature_Simplified'] = stable_df['Feature'].apply(simplify_feature_name)
             sns.barplot(x='Feature_Simplified', y='Mean_Coefficient', data=stable_df, palette='viridis')
             plt.title(f"Stable LASSO Features for {target}", fontsize=16)
             plt.ylabel("Average Coefficient", fontsize=12)
             plt.xlabel("")
             plt.xticks(rotation=90, ha='center', fontsize=10)
             plt.tight_layout()
             plt.savefig(os.path.join(FIGURES_DIR, f"Stable_LASSO_Features_{target}.png"), dpi=300)
             plt.show()
else:
    print("Skipping LASSO feature selection due to empty training data.")

```

# Section IV: OLS Model Analysis and Evaluation

```python
# --- Helper function for OLS and Evaluation ---
def run_ols_and_evaluate(formula, train_data, test_data, target_variable):
    """Fits OLS on train data, evaluates on test data."""
    results = {}
    try:
        # Fit model on training data
        model = ols(formula, data=train_data).fit()
        
        # Predict on test data
        X_test_model = sm.add_constant(test_data[model.params.index[1:]], has_constant='add') # Get features used in model
        y_pred_test = model.predict(X_test_model)
        y_true_test = test_data[target_variable]
        
        # Calculate test set metrics
        results['R2_test'] = r2_score(y_true_test, y_pred_test)
        results['RMSE_test'] = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
        results['MAE_test'] = mean_absolute_error(y_true_test, y_pred_test)
        results['AIC_train'] = model.aic # AIC is calculated on training data fit
        results['N_obs_test'] = len(y_true_test)
        results['y_true_test'] = y_true_test
        results['y_pred_test'] = y_pred_test
        
    except Exception as e:
        print(f"Error fitting/evaluating model with formula '{formula}': {e}")
        results = {metric: np.nan for metric in ['R2_test', 'RMSE_test', 'MAE_test', 'AIC_train', 'N_obs_test']}
        results['y_true_test'] = None
        results['y_pred_test'] = None
        
    return results

# --- Main Analysis Function ---
def perform_subscore_analysis(
    train_data, 
    test_data,
    mdsupdrs_vars, 
    y_vars, # delta_ledd targets
    basic_covariates, # Processed: Target_L_R_binary, Age_DBS_On, No_Leads_numeric, Sex_binary
    wb_fa_covariates, # basic + FA_15
    stable_features_per_target, # Dict: {'delta_ledd1': [stable_feats], ...}
    tract_features, # List of tract column names
    output_dir
):
    """
    Runs the full analysis loop for each subscore and model type.
    """
    all_results_summary = []
    all_results_predictions = {} # Store predictions for plots

    # Abbreviate MDSUPDRS variables for labeling
    mdsupdrs_abbrev = {
        'MDSUPDRSIIIpre_Percent_TOTAL_V': 'Total',
        'MDSUPDRSIIIpre_Percent_BRADY': 'Brady',
        'MDSUPDRSIIIpre_Percent_TREMOR': 'Tremor',
        'MDSUPDRSIIIpre_Percent_AXIAL': 'Axial'
    }
    dep_var_labels = { 'delta_ledd1': '0-3 mo.', 'delta_ledd2': '3-6 mo.', 'delta_ledd3': '6-12 mo.' }

    for mds_var in mdsupdrs_vars:
        mds_label = mdsupdrs_abbrev.get(mds_var, mds_var)
        print(f"\n===== Analyzing Subscore: {mds_label} ({mds_var}) =====")
        all_results_predictions[mds_label] = {}
        
        for y_var in y_vars:
            y_label = dep_var_labels.get(y_var, y_var)
            print(f"\n--- Target Timepoint: {y_label} ({y_var}) ---")
            all_results_predictions[mds_label][y_label] = {}

            # --- Model a: Clinical Only ---
            formula_a = f"{y_var} ~ {mds_var} + {' + '.join(basic_covariates)}"
            print(f"Model A Formula: {formula_a}")
            results_a = run_ols_and_evaluate(formula_a, train_data, test_data, y_var)
            all_results_summary.append({'Subscore': mds_label, 'Time Point': y_label, 'Model': 'Clinical only', **results_a})
            all_results_predictions[mds_label][y_label]['Clinical only'] = (results_a['y_true_test'], results_a['y_pred_test'])

            # --- Model b: Clinical + Whole Brain FA ---
            formula_b = f"{y_var} ~ {mds_var} + {' + '.join(wb_fa_covariates)}"
            print(f"Model B Formula: {formula_b}")
            results_b = run_ols_and_evaluate(formula_b, train_data, test_data, y_var)
            all_results_summary.append({'Subscore': mds_label, 'Time Point': y_label, 'Model': 'Clinical + wb-FA', **results_b})
            all_results_predictions[mds_label][y_label]['Clinical + wb-FA'] = (results_b['y_true_test'], results_b['y_pred_test'])

            # --- Model c: Clinical + Tract FA ---
            stable_tracts_for_target = [f for f in stable_features_per_target.get(y_var, []) if f in tract_features]
            if stable_tracts_for_target:
                model_c_features = basic_covariates + stable_tracts_for_target
                formula_c = f"{y_var} ~ {mds_var} + {' + '.join(model_c_features)}"
                print(f"Model C Formula: {formula_c}")
                results_c = run_ols_and_evaluate(formula_c, train_data, test_data, y_var)
                all_results_summary.append({'Subscore': mds_label, 'Time Point': y_label, 'Model': 'Clinical + tractFA', **results_c})
                all_results_predictions[mds_label][y_label]['Clinical + tractFA'] = (results_c['y_true_test'], results_c['y_pred_test'])
            else:
                print(f"Skipping Model C for {y_var} as no stable tract features were identified.")
                all_results_summary.append({'Subscore': mds_label, 'Time Point': y_label, 'Model': 'Clinical + tractFA', 
                                            'R2_test': np.nan, 'RMSE_test': np.nan, 'MAE_test': np.nan, 'AIC_train': np.nan, 'N_obs_test': 0})
                all_results_predictions[mds_label][y_label]['Clinical + tractFA'] = (None, None)


    # --- Generate Plots and Tables ---
    df_summary = pd.DataFrame(all_results_summary).drop(columns=['y_true_test', 'y_pred_test']) # Drop prediction arrays from summary table
    
    # Round numeric columns for display
    numeric_cols = ['R2_test', 'RMSE_test', 'MAE_test', 'AIC_train']
    for col in numeric_cols:
         if col in df_summary.columns:
              df_summary[col] = df_summary[col].round(3)

    # Plot Adjusted R-squared comparison
    model_order = ['Clinical only', 'Clinical + wb-FA', 'Clinical + tractFA']
    time_point_order = [dep_var_labels[y] for y in y_vars]
    colors = ['#1f77b4', '#9467bd', '#c5b0d5'] # Blue/Purple theme

    for mds_label in df_summary['Subscore'].unique():
        df_subscore = df_summary[df_summary['Subscore'] == mds_label]
        
        # R-squared Plot
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df_subscore, x='Model', y='R2_test', hue='Time Point', 
                    order=model_order, hue_order=time_point_order, palette=colors, edgecolor='black')
        plt.ylabel(f'Test Set R-squared ({mds_label})', fontsize=14)
        plt.xlabel('Model Type', fontsize=14)
        plt.title(f'Model Comparison: Test Set R-squared for {mds_label} Subscore', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Time Point', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"R2_Test_{mds_label}_Model_Comparison.png"), dpi=300)
        plt.show()

        # Predicted vs Actual Plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
        fig.suptitle(f'Predicted vs Actual ({mds_label} Subscore Models - Test Set)', fontsize=16, y=1.02)
        for i, y_label_plot in enumerate(time_point_order):
             ax = axes[i]
             for model_name in model_order:
                  y_true, y_pred = all_results_predictions[mds_label][y_label_plot].get(model_name, (None, None))
                  if y_true is not None and y_pred is not None:
                       ax.scatter(y_true, y_pred, alpha=0.6, label=model_name)
             
             # Add identity line
             lims = [
                  np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                  np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
             ]
             ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label='Ideal')
             ax.set_aspect('equal')
             ax.set_xlim(lims)
             ax.set_ylim(lims)
             ax.set_title(y_label_plot, fontsize=14)
             ax.set_xlabel("Actual Values", fontsize=12)
             if i == 0:
                  ax.set_ylabel("Predicted Values", fontsize=12)
             ax.legend(fontsize=10)
             ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
        plt.savefig(os.path.join(output_dir, f"Pred_vs_Actual_{mds_label}.png"), dpi=300)
        plt.show()

        # Display and Save Summary Table for this subscore
        print(f"\n=== Summary Table: {mds_label} Subscore ===\n")
        df_subscore_display = df_subscore[['Model', 'Time Point', 'R2_test', 'RMSE_test', 'MAE_test', 'AIC_train', 'N_obs_test']]
        display(df_subscore_display.style.set_caption(f"{mds_label} Model Results").format(precision=3))
        
        # Save to Excel
        excel_filename = os.path.join(output_dir, f"{mds_label}_model_summary.xlsx")
        try:
            df_subscore_display.to_excel(excel_filename, index=False)
            print(f"Saved model summary table to {excel_filename}")
        except Exception as e:
            print(f"Error saving Excel summary table {excel_filename}: {e}")
            
        # Save styled table as image (optional)
        # try:
        #     table_image_filename = os.path.join(output_dir, f"{mds_label}_model_summary.png")
        #     dfi.export(df_subscore_display.style.format(precision=3), table_image_filename)
        #     print(f"Saved model summary table image to {table_image_filename}")
        # except Exception as e:
        #     print(f"Error saving model summary table image {table_image_filename}: {e}")

    return df_summary, all_results_predictions

# --- Run the Analysis ---
if not train_df.empty and not test_df.empty:
    analysis_summary_df, analysis_predictions = perform_subscore_analysis(
        train_data=train_df,
        test_data=test_df,
        mdsupdrs_vars=mdsupdrs_predictors,
        y_vars=delta_ledd_targets,
        basic_covariates=final_basic_covariates,
        wb_fa_covariates=final_wb_fa_covariates,
        stable_features_per_target=stable_features_per_ledd,
        tract_features=tract_feature_cols,
        output_dir=FIGURES_DIR # Save plots to figures subdir
    )
else:
    print("Skipping OLS analysis due to empty train/test data.")
    analysis_summary_df = pd.DataFrame()
    analysis_predictions = {}

```

# Section V: Brain Visualization

```python
# --- Brain Visualization Function ---
def plot_top_tracts_single_row(top_tract_names, top_tract_coefficients, delta_id, output_dir):
    """
    Plots the top tracts based on LASSO coefficients in a single row (sagittal, coronal, axial).
    Uses 'coolwarm' colormap.
    """
    print(f"\n--- Plotting Brain Map for: {delta_id} ---")
    
    # Define the mapping from tract names to intensity values based on the atlas
    tract_mapping = {
        'ATR L': 1, 'ATR R': 2, 'CST L': 3, 'CST R': 4, 'CGC L': 5, 'CGC R': 6, 
        'CGH L': 7, 'CGH R': 8, 'FMA': 9, 'FMI': 10, 'IFOF L': 11, 'IFOF R': 12, 
        'ILF L': 13, 'ILF R': 14, 'SLF L': 15, 'SLF R': 16, 'SLFT L': 17, 'SLFT R': 18, 
        'UF L': 19, 'UF R': 20
        # Add PTR L/R if they exist in your atlas and are needed
    }

    # Load atlas and template
    try:
        atlas_img = nib.load(ATLAS_FILE)
        template = datasets.load_mni152_template()
    except FileNotFoundError as e:
        print(f"Error loading atlas or template: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading atlas/template: {e}")
        return

    # Simplify tract names from LASSO results
    top_tracts_simplified = [simplify_tract_name(name) for name in top_tract_names]

    # Filter based on valid mapping and coefficients
    valid_indices = [i for i, tract in enumerate(top_tracts_simplified) if tract in tract_mapping]
    if not valid_indices:
        print(f"No valid tracts found in mapping for {delta_id}. Skipping plot.")
        return
        
    top_tracts_simplified = [top_tracts_simplified[i] for i in valid_indices]
    top_coefficients_filtered = [top_tract_coefficients[i] for i in valid_indices]

    # Create intensity mapping for plotting
    tract_label_mapping = {tract: i + 1 for i, tract in enumerate(top_tracts_simplified)}
    
    # Create combined tract image based on labels
    combined_tract_data = np.zeros(atlas_img.shape, dtype=np.int16)
    for tract, label in tract_label_mapping.items():
        intensity = tract_mapping[tract]
        tract_mask = (atlas_img.get_fdata() == intensity)
        combined_tract_data[tract_mask] = label
        
    combined_tract_img = nib.Nifti1Image(combined_tract_data, atlas_img.affine)
    resampled_tract_img = image.resample_to_img(combined_tract_img, template, interpolation='nearest')

    # Create colormap based on coefficients
    vmin, vmax = np.min(top_coefficients_filtered), np.max(top_coefficients_filtered)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_coeffs = cm.get_cmap('coolwarm') # Use coolwarm
    
    # Map labels to colors based on coefficients
    label_to_color = {tract_label_mapping[tract]: cmap_coeffs(norm(coeff)) 
                      for tract, coeff in zip(top_tracts_simplified, top_coefficients_filtered)}
    
    # Create discrete colormap for plotting ROIs
    plot_colors = [label_to_color.get(i, (0,0,0,0)) for i in range(1, len(top_tracts_simplified) + 1)] # Use transparent for missing labels
    custom_cmap_roi = ListedColormap(plot_colors)
    
    # Define cut coordinates (adjust as needed)
    cut_coords = {'x': 0, 'y': -10, 'z': 10} 
    display_modes = ['x', 'y', 'z'] # sagittal, coronal, axial

    # Create figure (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

    for i, display_mode in enumerate(display_modes):
        ax = axes[i]
        plotting.plot_roi(
            resampled_tract_img,
            bg_img=template,
            display_mode=display_mode,
            cut_coords=[cut_coords[display_mode]],
            axes=ax,
            cmap=custom_cmap_roi,
            colorbar=False, 
            threshold=0.5, 
            black_bg=False,
            annotate=False,
            alpha=0.8 # Make tracts slightly transparent
        )
        ax.set_title(f"{display_mode.upper()}-view", fontsize=14) # Add view title

    plt.tight_layout()
    
    # Save the main plot
    output_image_path = os.path.join(output_dir, f'{delta_id}_tract_visualization_coolwarm.png')
    plt.savefig(output_image_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig) # Close plot figure
    print(f"Saved brain map: {output_image_path}")

    # Create and save the colorbar separately
    fig_colorbar, ax_colorbar = plt.subplots(figsize=(1.5, 6))
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_coeffs), cax=ax_colorbar, orientation='vertical')
    cbar.set_label('LASSO Coefficient', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    colorbar_image_path = os.path.join(output_dir, f'{delta_id}_colorbar_coolwarm.png')
    plt.savefig(colorbar_image_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig_colorbar) # Close colorbar figure
    print(f"Saved colorbar: {colorbar_image_path}")


# --- Generate Brain Maps ---
if stable_coeffs_dfs:
     for target, df_coeffs in stable_coeffs_dfs.items():
          # Filter for tract features only for visualization
          tract_coeffs_df = df_coeffs[df_coeffs['Feature'].isin(tract_feature_cols)]
          if not tract_coeffs_df.empty:
               plot_top_tracts_single_row(
                    top_tract_names=tract_coeffs_df['Feature'].tolist(), 
                    top_tract_coefficients=tract_coeffs_df['Mean_Coefficient'].tolist(), 
                    delta_id=target,
                    output_dir=FIGURES_DIR
               )
          else:
               print(f"No stable tract features found for {target} to visualize.")
else:
     print("Skipping brain visualization as no stable features were identified.")

print("\nAnalysis Complete.")
