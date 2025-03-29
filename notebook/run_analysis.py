# Extracted Python code from FAproject.md

# Section I: Setup and Data Preparation
### Import libraries.
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
# Removed IPython display as it's not needed for script execution
# from IPython.display import display 
import dataframe_image as dfi # Optional: for exporting styled tables as images

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning) # Suppress UserWarnings often from statsmodels/seaborn

# Plotting defaults
rcParams['font.family'] = 'Helvetica'
sns.set(style="white", rc={'axes.grid': False, 'figure.dpi': 150}) 

# --- Helper Functions ---
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
        elif value == 1: # Handle cases where it might already be numeric
            return 1
        elif value == 0:
            return 0
        return np.nan # Return NaN for other cases

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
            value_lower = value.lower().strip()
            if value_lower.startswith('m') or 'male' in value_lower:
                return 1
            elif value_lower.startswith('f') or 'female' in value_lower:
                return 0
        return np.nan # Return NaN for non-string or unmapped values

    return sex_series.apply(map_sex)

def process_no_leads(no_leads_series):
    """
    Clean the 'No_Leads' column:
    - Convert 'BI' to 'Bi'
    - Keep only 'Bi', 'R', 'L'
    - Assign numerical values: 'Bi' = 2, 'R' = 1, 'L' = 1 (Treat R/L as unilateral)
    - Assign 0 for others/NaN
    """
    def map_no_leads(value):
        if isinstance(value, str):
            value = value.strip().upper()
            if value == 'BI':
                return 2 # Bilateral
            elif value == 'R':
                return 1 # Unilateral Right
            elif value == 'L':
                return 1 # Unilateral Left
        # Consider returning NaN if 0 is not meaningful or if you want to drop these cases
        return 0 # Assume 0 or handle as NaN if appropriate 

    return no_leads_series.apply(map_no_leads)

def simplify_feature_name(feature_name):
    """
    Simplify feature names for better readability in plots.
    Handles potential dummy variable suffixes created by get_dummies.
    """
    # First, handle potential dummy variable suffixes like _1.0, _2.0 etc.
    name = re.sub(r'_(\d+(\.\d+)?)$', '', feature_name) # Remove suffix like _1.0
    name = re.sub(r'\[T\.(\d+(\.\d+)?)\]$', '', name) # Remove statsmodels suffix like [T.1.0]
    
    # Apply original simplifications
    name = name.replace('_fa', '').replace('_', ' ')
    name = name.replace('Sex binary', 'Sex').replace('No Leads numeric', 'No Leads').replace('Target L R binary', 'Target')
    name = name.replace('MDSUPDRSIIIpre Percent TOTAL V', 'MDSUPDRS III Total')
    name = name.replace('MDSUPDRSIIIpre Percent BRADY', 'MDSUPDRS III Brady')
    name = name.replace('MDSUPDRSIIIpre Percent TREMOR', 'MDSUPDRS III Tremor')
    name = name.replace('MDSUPDRSIIIpre Percent AXIAL', 'MDSUPDRS III Axial')
    name = name.replace('Age DBS On', 'Age')
    # Add more specific replacements if needed
    return name.strip()

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

### Load and Combine Tract FA Data
# (Code remains the same as before)
# Regex to match the filenames
pattern = r'sub-PDa\d{3}_ses-01_dwi_space-RASMM_model-probCSD_algo-reco80_desc-profiles_dwi.csv'
try:
    afq_files = [os.path.join(MRTRIX_CSV_DIR, f) for f in os.listdir(MRTRIX_CSV_DIR) if re.match(pattern, f)]
except FileNotFoundError:
    print(f"Error: Directory not found - {MRTRIX_CSV_DIR}")
    afq_files = []
if not afq_files:
     print(f"Warning: No AFQ profile CSV files found matching the pattern in {MRTRIX_CSV_DIR}")
     combined_fa_data = pd.DataFrame(columns=['nodeID', 'dti_fa', 'tractID', 'subject_id'])
else:
    fa_data_list = []
    for file in afq_files:
        try:
            df = pd.read_csv(file)
            subject_id = os.path.basename(file).split('_')[0]
            df['subject_id'] = subject_id
            fa_data_list.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    if fa_data_list:
        combined_fa_data = pd.concat(fa_data_list, ignore_index=True)
        print(f"Loaded data from {len(fa_data_list)} subject CSV files.")
    else:
        print("No valid FA data loaded.")
        combined_fa_data = pd.DataFrame(columns=['nodeID', 'dti_fa', 'tractID', 'subject_id'])
print("Combined FA Data Head:")
print(combined_fa_data.head()) # Use print instead of display

### Load and Prepare Clinical Outcome Data
# (Code remains the same as before)
try:
    outcome_data_raw = pd.read_excel(OUTCOME_FILE)
    print(f"Loaded outcome data from: {OUTCOME_FILE}")
except FileNotFoundError:
    print(f"Error: Outcome file not found - {OUTCOME_FILE}")
    outcome_data_raw = pd.DataFrame()
if not outcome_data_raw.empty:
    outcome_data = outcome_data_raw.rename(columns={
        'PTID_Retro_Clin': 'subject_id', 
        'LEDDpost1_Delta': 'delta_ledd1', 
        'LEDDpost2_Delta': 'delta_ledd2', 
        'LEDDpost3_Delta': 'delta_ledd3'
    })
    if 'subject_id' in outcome_data.columns:
        outcome_data['subject_id'] = outcome_data['subject_id'].astype(str)
        outcome_data['subject_id'] = outcome_data['subject_id'].apply(lambda x: f"sub-{x}" if not x.startswith('sub-') else x)
        outcome_data = outcome_data.drop_duplicates(subset=['subject_id'])
        for col in ['delta_ledd1', 'delta_ledd2', 'delta_ledd3']:
             if col in outcome_data.columns:
                  outcome_data[col] = pd.to_numeric(outcome_data[col], errors='coerce')
        for col in outcome_data.select_dtypes(include='object').columns:
             outcome_data[col] = outcome_data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        print("Outcome Data Head after cleaning:")
        print(outcome_data.head()) # Use print instead of display
    else:
        print("Warning: 'PTID_Retro_Clin' column not found in outcome data.")
        outcome_data = pd.DataFrame()
else:
     outcome_data = pd.DataFrame()

### Merge FA Data and Clinical Data
# (Code remains the same as before)
if not combined_fa_data.empty and not outcome_data.empty:
    data_merged = pd.merge(combined_fa_data, outcome_data, on='subject_id', how='inner')
    print("Merged FA and Outcome Data Head:")
    print(data_merged.head()) # Use print instead of display
    print(f"Number of rows after merge: {len(data_merged)}")
else:
    print("Skipping merge due to empty FA or outcome data.")
    data_merged = pd.DataFrame() 

### Aggregate Tract FA Values
# (Code remains the same as before)
if not data_merged.empty:
    average_fa_per_subject_tract = data_merged.groupby(['subject_id', 'tractID'])['dti_fa'].mean().reset_index()
    clinical_data_unique = data_merged.drop(columns=['nodeID', 'dti_fa', 'tractID']).drop_duplicates(subset=['subject_id'])
    tract_specific_data = pd.merge(average_fa_per_subject_tract, clinical_data_unique, on='subject_id', how='inner')
    print("Tract Specific Data Head:")
    print(tract_specific_data.head()) # Use print instead of display
    print(f"Number of rows in tract_specific_data: {len(tract_specific_data)}")
    tract_specific_data.to_csv(os.path.join(RESULTS_DIR, 'tract_specific_data_intermediate.csv'), index=False)
else:
    print("Skipping tract aggregation due to empty merged data.")
    tract_specific_data = pd.DataFrame()

### Filter Tracts Based on Presence
# (Code remains the same as before)
if not tract_specific_data.empty:
    total_subjects = tract_specific_data['subject_id'].nunique()
    min_subjects_threshold = 100
    tract_counts = tract_specific_data.groupby('tractID')['subject_id'].nunique()
    valid_tracts = tract_counts[tract_counts >= min_subjects_threshold].index
    relevant_tract_data = tract_specific_data[tract_specific_data['tractID'].isin(valid_tracts)].copy()
    print(f"Filtered tracts present in >= {min_subjects_threshold} subjects. Kept {len(valid_tracts)} tracts.")
    print("Relevant Tract Data Head:")
    print(relevant_tract_data.head()) # Use print instead of display
    relevant_tract_data.to_csv(os.path.join(RESULTS_DIR, 'relevant_tract_data_intermediate.csv'), index=False)
else:
    print("Skipping tract filtering due to empty tract_specific_data.")
    relevant_tract_data = pd.DataFrame()
    valid_tracts = []

### Create Final Patient DataFrame (Pivot Tracts) & Preprocess for Modeling
if not relevant_tract_data.empty:
    pivot_df_tracts = relevant_tract_data.pivot_table(index='subject_id', columns='tractID', values='dti_fa')
    clinical_data_unique = relevant_tract_data.drop(columns=['tractID', 'dti_fa']).drop_duplicates(subset=['subject_id']).set_index('subject_id')
    patient_df_prelim = clinical_data_unique.join(pivot_df_tracts)
    
    # Load and merge whole brain FA data
    try:
        wholebrain_df = pd.read_csv(WB_FA_FILE)
        if 'ID' in wholebrain_df.columns:
             wholebrain_df['subject_id'] = wholebrain_df['ID'].astype(str).str.replace('_fa.nii.gz', '', regex=False)
             wholebrain_df['subject_id'] = wholebrain_df['subject_id'].apply(lambda x: f"sub-{x}" if not x.startswith('sub-') else x)
             wholebrain_df = wholebrain_df.drop(columns=['ID'])
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
    patient_df_processed = patient_df_merged.copy()
    
    # Process binary/numeric clinical covariates
    patient_df_processed['Target_L_R_binary'] = process_target_lr(patient_df_processed['Target_L_R'])
    patient_df_processed['Sex_binary'] = process_sex(patient_df_processed['Sex'])
    patient_df_processed['No_Leads_numeric'] = process_no_leads(patient_df_processed['No_Leads'])

    # Define column groups
    delta_ledd_targets = ['delta_ledd1', 'delta_ledd2', 'delta_ledd3']
    mdsupdrs_predictors_raw = [ # Keep original names for dummification
        'MDSUPDRSIIIpre_Percent_TOTAL_V',
        'MDSUPDRSIIIpre_Percent_BRADY',
        'MDSUPDRSIIIpre_Percent_TREMOR',
        'MDSUPDRSIIIpre_Percent_AXIAL'
    ]
    basic_clinical_covariates_processed = ['Target_L_R_binary', 'Age_DBS_On', 'No_Leads_numeric', 'Sex_binary']
    wb_fa_predictor = ['FA_15'] 
    
    # Identify tract columns
    all_other_cols = set(delta_ledd_targets + mdsupdrs_predictors_raw + basic_clinical_covariates_raw + wb_fa_predictor + ['subject_id', 'Target_L_R_binary', 'Sex_binary', 'No_Leads_numeric']) 
    all_other_cols.update([col for col in patient_df_processed.columns if not ('_fa' in col or col in basic_clinical_covariates_raw or col in mdsupdrs_predictors_raw or col in delta_ledd_targets or col == 'subject_id')])
    tract_feature_cols = [col for col in patient_df_processed.columns if col not in all_other_cols and '_fa' in col] 

    # --- Create Dummy Variables for MDS-UPDRS ---
    # Ensure predictors are categorical before dummification if they aren't already
    for col in mdsupdrs_predictors_raw:
        if col in patient_df_processed.columns and pd.api.types.is_numeric_dtype(patient_df_processed[col]):
             # Optional: Convert to string or category if needed, depends on data nature
             # patient_df_processed[col] = patient_df_processed[col].astype(str) 
             pass # Assuming they are numeric and should be treated as such by OLS/LASSO unless specified otherwise

    # Define final feature sets (using processed clinical vars)
    final_basic_covariates = [col for col in basic_clinical_covariates_processed if col in patient_df_processed.columns]
    final_wb_fa_covariates = final_basic_covariates + [col for col in wb_fa_predictor if col in patient_df_processed.columns]
    final_wb_fa_covariates = sorted(list(set(final_wb_fa_covariates))) # Unique and sorted

    # Define columns needed for dropping NaNs (targets + all potential predictors/covariates)
    essential_cols = (delta_ledd_targets + mdsupdrs_predictors_raw + 
                      final_basic_covariates + final_wb_fa_covariates + tract_feature_cols)
    essential_cols = [col for col in essential_cols if col in patient_df_processed.columns]
    essential_cols = sorted(list(set(essential_cols)))

    patient_df_final = patient_df_processed.dropna(subset=essential_cols).copy()
    
    print(f"Final dataset size after dropping NaNs in essential columns: {patient_df_final.shape[0]} subjects.")

    # Save the final processed dataframe
    patient_df_final.reset_index().to_csv(os.path.join(RESULTS_DIR, 'patient_df_final_processed.csv'), index=False)
    
    print("Final Processed Patient Data Head:")
    print(patient_df_final.head()) # Use print instead of display

else:
    print("Skipping final patient dataframe creation.")
    patient_df_final = pd.DataFrame()
    tract_feature_cols = []
    final_basic_covariates = []
    final_wb_fa_covariates = []
    mdsupdrs_predictors_raw = []
    delta_ledd_targets = []

# Section II: Train/Test Split
if not patient_df_final.empty:
    # Define features (X) and targets (Y) 
    # Features include processed clinical, wb_fa, tracts, AND raw mdsupdrs (OLS will handle them)
    feature_cols = final_basic_covariates + final_wb_fa_covariates + tract_feature_cols + mdsupdrs_predictors_raw
    feature_cols = [col for col in feature_cols if col in patient_df_final.columns]
    feature_cols = sorted(list(set(feature_cols))) 
    
    X = patient_df_final[feature_cols]
    Y_cols = [col for col in delta_ledd_targets if col in patient_df_final.columns]
    Y = patient_df_final[Y_cols] 

    # Stratify if possible
    stratify_col_data = None
    if 'Target_L_R_binary' in patient_df_final.columns and patient_df_final['Target_L_R_binary'].nunique() > 1:
         min_class_count = patient_df_final['Target_L_R_binary'].value_counts().min()
         if min_class_count >= 2:
              stratify_col_data = patient_df_final['Target_L_R_binary']
              print("Stratifying train/test split by 'Target_L_R_binary'.")
         else:
              print("Warning: Not enough samples in each 'Target_L_R_binary' class for stratification.")
    else:
         print("Stratification column 'Target_L_R_binary' not suitable or not found.")

    # Perform the split on X and Y separately
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_SEED,
        stratify=stratify_col_data # Pass the Series for stratification
    )

    # Combine back into train_df and test_df
    train_df = pd.concat([X_train, Y_train], axis=1)
    test_df = pd.concat([X_test, Y_test], axis=1)

    print(f"Training set size: {train_df.shape[0]}")
    print(f"Test set size: {test_df.shape[0]}")
else:
    print("Skipping train/test split due to empty data.")
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

# Section III: Feature Selection (LASSO)
# --- LASSO Feature Selection Function ---
def run_lasso_feature_selection(target_variable, predictor_variable, train_data, feature_names, 
                                alpha, n_iterations, selection_threshold):
    """
    Runs LASSO with bootstrapping on training data to find stable features for predicting a target,
    potentially including a primary predictor (like an MDS-UPDRS score).
    Uses StandardScaler for feature scaling.
    """
    print(f"\n--- Running LASSO for target: {target_variable} (Predictor: {predictor_variable}) ---")
    
    # Prepare Target Variable and Features from training data
    y_train = train_data[target_variable].astype(float).values.ravel()
    
    # Define features for LASSO: all potential features EXCEPT the target variable itself.
    # The predictor_variable IS included here as LASSO will decide its importance.
    current_feature_names = [f for f in feature_names if f != target_variable and f in train_data.columns]
    X_train_full = train_data[current_feature_names].values 
    
    n_samples = X_train_full.shape[0]
    
    if n_samples == 0:
        print("Warning: Training data is empty. Skipping LASSO.")
        return [], pd.DataFrame()

    selected_features_indices_list = []
    coefficients_list = []
    
    # Scale the full training feature set once before bootstrapping
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)

    for i in range(n_iterations):
        # Bootstrap resampling of the scaled training data indices
        indices = resample(np.arange(n_samples), random_state=RANDOM_SEED + i)
        X_resampled_scaled = X_train_full_scaled[indices]
        y_resampled = y_train[indices]
        
        # Lasso Regression
        lasso = Lasso(alpha=alpha, max_iter=10000, random_state=RANDOM_SEED, tol=1e-3) # Adjust tol if needed
        lasso.fit(X_resampled_scaled, y_resampled)
        
        # Store coefficients and selected features
        coefficients_list.append(lasso.coef_)
        non_zero_indices = np.where(np.abs(lasso.coef_) > 1e-6)[0] # Use tolerance for near-zero
        selected_features_indices_list.append(non_zero_indices)
        
    # --- Feature Stability Analysis ---
    feature_counter = Counter()
    for indices in selected_features_indices_list:
        feature_counter.update(indices)
        
    threshold_count = n_iterations * selection_threshold
    stable_feature_indices = [idx for idx, count in feature_counter.items() if count >= threshold_count]
    
    if not stable_feature_indices:
        print(f"No features met the {selection_threshold*100}% stability threshold for {target_variable} predicted by {predictor_variable}.")
        return [], pd.DataFrame()

    stable_feature_names = [current_feature_names[i] for i in stable_feature_indices]
    
    # Compute average coefficients for stable features over all iterations
    coefficients_array = np.array(coefficients_list)
    mean_coefficients_all = np.nanmean(coefficients_array, axis=0)
    stable_mean_coefficients = mean_coefficients_all[stable_feature_indices]

    df_stable_coeffs = pd.DataFrame({
        'Feature': stable_feature_names,
        'Mean_Coefficient': stable_mean_coefficients
    })
    df_stable_coeffs['Abs_Coefficient'] = df_stable_coeffs['Mean_Coefficient'].abs()
    df_stable_coeffs.sort_values(by='Abs_Coefficient', ascending=False, inplace=True)

    print(f"Identified {len(stable_feature_names)} stable features for {target_variable} (Predictor: {predictor_variable}, threshold={selection_threshold*100}%).")
    
    return stable_feature_names, df_stable_coeffs

# --- Run LASSO for each combination of delta_ledd target and MDS-UPDRS predictor ---
stable_features_per_model = {} # Nested dict: {mds_var: {ledd_var: [features]}}
stable_coeffs_dfs_per_model = {} # Nested dict: {mds_var: {ledd_var: DataFrame}}

if not train_df.empty:
    # Define all potential features for LASSO (excluding targets)
    # Include processed clinical, wb_fa, tracts, AND raw mdsupdrs (as potential predictors/covariates)
    lasso_feature_pool = final_basic_covariates + final_wb_fa_covariates + tract_feature_cols + mdsupdrs_predictors_raw
    lasso_feature_pool = [f for f in lasso_feature_pool if f not in delta_ledd_targets and f in train_df.columns] 
    lasso_feature_pool = sorted(list(set(lasso_feature_pool)))

    for mds_pred in mdsupdrs_predictors_raw: # Use raw names here
         if mds_pred not in train_df.columns: continue # Skip if predictor missing
         stable_features_per_model[mds_pred] = {}
         stable_coeffs_dfs_per_model[mds_pred] = {}
         for target in delta_ledd_targets:
              if target not in train_df.columns: continue # Skip if target missing
              
              # Features for this run = pool - target
              current_lasso_features = [f for f in lasso_feature_pool if f != target]
              
              stable_names, stable_df = run_lasso_feature_selection(
                   target_variable=target,
                   predictor_variable=mds_pred, # Pass the main predictor name
                   train_data=train_df,
                   feature_names=current_lasso_features, 
                   alpha=LASSO_ALPHA,
                   n_iterations=LASSO_ITERATIONS,
                   selection_threshold=LASSO_SELECTION_THRESHOLD
              )
              stable_features_per_model[mds_pred][target] = stable_names
              stable_coeffs_dfs_per_model[mds_pred][target] = stable_df
              
              # Plot stable coefficients
              if not stable_df.empty:
                   plt.figure(figsize=(12, 6))
                   stable_df['Feature_Simplified'] = stable_df['Feature'].apply(simplify_feature_name)
                   sns.barplot(x='Feature_Simplified', y='Mean_Coefficient', data=stable_df, palette='viridis')
                   plt.title(f"Stable LASSO Features for {target} (Predictor: {simplify_feature_name(mds_pred)})", fontsize=16)
                   plt.ylabel("Average Coefficient", fontsize=12)
                   plt.xlabel("")
                   plt.xticks(rotation=90, ha='center', fontsize=10)
                   plt.tight_layout()
                   plt.savefig(os.path.join(FIGURES_DIR, f"Stable_LASSO_Features_{target}_pred_{mds_pred}.png"), dpi=300)
                   plt.close() # Close figure
else:
    print("Skipping LASSO feature selection due to empty training data.")

# Section IV: OLS Model Analysis and Evaluation
# --- Helper function for OLS and Evaluation ---
def run_ols_and_evaluate(formula, train_data, test_data, target_variable):
    """Fits OLS on train data, evaluates on test data."""
    results = {}
    try:
        # Ensure target exists
        if target_variable not in train_data.columns or target_variable not in test_data.columns:
             raise ValueError(f"Target variable '{target_variable}' not found in train or test data.")
             
        # Extract predictor names from formula (handle potential interactions later if needed)
        # Simple approach: find all words that are columns in train_data
        formula_vars = re.findall(r'\b\w+\b', formula)
        predictor_cols = [v for v in formula_vars if v in train_data.columns and v != target_variable]
        
        # Ensure all predictor columns exist in both train and test data
        missing_train = [col for col in predictor_cols if col not in train_data.columns]
        missing_test = [col for col in predictor_cols if col not in test_data.columns]
        if missing_train or missing_test:
             raise ValueError(f"Missing columns. Train: {missing_train}, Test: {missing_test}")

        # Prepare data for statsmodels (it handles dummy creation internally if needed)
        y_train = train_data[target_variable]
        X_train = train_data[predictor_cols]
        X_test = test_data[predictor_cols]
        y_true_test = test_data[target_variable]

        # Add constant for intercept
        X_train_sm = sm.add_constant(X_train, has_constant='add')
        X_test_sm = sm.add_constant(X_test, has_constant='add')

        # Fit model on training data
        model = sm.OLS(y_train, X_train_sm).fit()
        
        # Predict on test data
        y_pred_test = model.predict(X_test_sm)
        
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
# (Code remains largely the same, but uses the modified run_ols_and_evaluate)
def perform_subscore_analysis(
    train_data, 
    test_data,
    mdsupdrs_vars, # Raw predictor names
    y_vars, # delta_ledd targets
    basic_covariates, # Processed: Target_L_R_binary, Age_DBS_On, No_Leads_numeric, Sex_binary
    wb_fa_covariates, # basic + FA_15
    stable_features_per_model, # Dict: {mds_var: {ledd_var: [features]}}
    tract_features, # List of tract column names
    output_dir
):
    """
    Runs the full analysis loop for each subscore and model type.
    """
    all_results_summary = []
    all_results_predictions = {} # Store predictions for plots

    mdsupdrs_abbrev = {
        'MDSUPDRSIIIpre_Percent_TOTAL_V': 'Total',
        'MDSUPDRSIIIpre_Percent_BRADY': 'Brady',
        'MDSUPDRSIIIpre_Percent_TREMOR': 'Tremor',
        'MDSUPDRSIIIpre_Percent_AXIAL': 'Axial'
    }
    dep_var_labels = { 'delta_ledd1': '0-3 mo.', 'delta_ledd2': '3-6 mo.', 'delta_ledd3': '6-12 mo.' }

    for mds_var in mdsupdrs_vars:
        if mds_var not in train_data.columns: continue # Skip if predictor missing
        mds_label = mdsupdrs_abbrev.get(mds_var, mds_var)
        print(f"\n===== Analyzing Subscore: {mds_label} ({mds_var}) =====")
        all_results_predictions[mds_label] = {}
        
        for y_var in y_vars:
            if y_var not in train_data.columns: continue # Skip if target missing
            y_label = dep_var_labels.get(y_var, y_var)
            print(f"\n--- Target Timepoint: {y_label} ({y_var}) ---")
            all_results_predictions[mds_label][y_label] = {}

            # Ensure covariates exist in data
            model_a_features = [f for f in basic_covariates if f in train_data.columns]
            model_b_features = [f for f in wb_fa_covariates if f in train_data.columns]

            # --- Model a: Clinical Only ---
            formula_a = f"`{y_var}` ~ `{mds_var}` + {' + '.join([f'`{f}`' for f in model_a_features])}" # Backticks for safety
            print(f"Model A Formula: {formula_a}")
            results_a = run_ols_and_evaluate(formula_a, train_data, test_data, y_var)
            all_results_summary.append({'Subscore': mds_label, 'Time Point': y_label, 'Model': 'Clinical only', **results_a})
            all_results_predictions[mds_label][y_label]['Clinical only'] = (results_a['y_true_test'], results_a['y_pred_test'])

            # --- Model b: Clinical + Whole Brain FA ---
            formula_b = f"`{y_var}` ~ `{mds_var}` + {' + '.join([f'`{f}`' for f in model_b_features])}"
            print(f"Model B Formula: {formula_b}")
            results_b = run_ols_and_evaluate(formula_b, train_data, test_data, y_var)
            all_results_summary.append({'Subscore': mds_label, 'Time Point': y_label, 'Model': 'Clinical + wb-FA', **results_b})
            all_results_predictions[mds_label][y_label]['Clinical + wb-FA'] = (results_b['y_true_test'], results_b['y_pred_test'])

            # --- Model c: Clinical + Tract FA ---
            stable_features_for_combo = stable_features_per_model.get(mds_var, {}).get(y_var, [])
            # Ensure stable features exist in data and are actual tract features
            stable_tracts_for_combo = [f for f in stable_features_for_combo if f in tract_features and f in train_data.columns]
            
            if stable_tracts_for_combo:
                model_c_feature_names = model_a_features + stable_tracts_for_combo 
                model_c_feature_names = sorted(list(set(model_c_feature_names))) # Unique, sorted
                formula_c = f"`{y_var}` ~ `{mds_var}` + {' + '.join([f'`{f}`' for f in model_c_feature_names])}"
                print(f"Model C Formula: {formula_c}")
                results_c = run_ols_and_evaluate(formula_c, train_data, test_data, y_var)
                all_results_summary.append({'Subscore': mds_label, 'Time Point': y_label, 'Model': 'Clinical + tractFA', **results_c})
                all_results_predictions[mds_label][y_label]['Clinical + tractFA'] = (results_c['y_true_test'], results_c['y_pred_test'])
            else:
                print(f"Skipping Model C for {y_var} predicting {mds_var} as no stable tract features were identified or valid.")
                all_results_summary.append({'Subscore': mds_label, 'Time Point': y_label, 'Model': 'Clinical + tractFA', 
                                            'R2_test': np.nan, 'RMSE_test': np.nan, 'MAE_test': np.nan, 'AIC_train': np.nan, 'N_obs_test': 0})
                all_results_predictions[mds_label][y_label]['Clinical + tractFA'] = (None, None)

    # --- Generate Plots and Tables ---
    # (Plotting code remains the same, using the results dataframe)
    df_summary = pd.DataFrame(all_results_summary).drop(columns=['y_true_test', 'y_pred_test']) 
    numeric_cols = ['R2_test', 'RMSE_test', 'MAE_test', 'AIC_train']
    for col in numeric_cols:
         if col in df_summary.columns:
              df_summary[col] = df_summary[col].round(3)

    model_order = ['Clinical only', 'Clinical + wb-FA', 'Clinical + tractFA']
    time_point_order = [dep_var_labels[y] for y in y_vars if y in train_data.columns] # Use only existing targets
    colors = ['#1f77b4', '#9467bd', '#c5b0d5'] 

    for mds_label in df_summary['Subscore'].unique():
        df_subscore = df_summary[df_summary['Subscore'] == mds_label].copy()
        if df_subscore.empty: continue
        df_subscore['Model'] = pd.Categorical(df_subscore['Model'], categories=model_order, ordered=True)
        df_subscore.sort_values('Model', inplace=True)
        
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
        plt.close() 

        # Predicted vs Actual Plots
        fig, axes = plt.subplots(1, len(time_point_order), figsize=(6 * len(time_point_order), 5), sharey=True, sharex=True)
        if len(time_point_order) == 1: axes = [axes] # Make iterable if only one subplot
        fig.suptitle(f'Predicted vs Actual ({mds_label} Subscore Models - Test Set)', fontsize=16, y=1.02)
        for i, y_label_plot in enumerate(time_point_order):
             ax = axes[i]
             has_data = False
             for model_name in model_order:
                  y_true, y_pred = all_results_predictions[mds_label][y_label_plot].get(model_name, (None, None))
                  if y_true is not None and y_pred is not None and len(y_true) > 0: 
                       ax.scatter(y_true, y_pred, alpha=0.6, label=model_name)
                       has_data = True
             
             if has_data:
                 all_vals = np.concatenate([vals for vals in [all_results_predictions[mds_label][y_label_plot][m][0] for m in model_order if all_results_predictions[mds_label][y_label_plot][m][0] is not None] + 
                                           [vals for vals in [all_results_predictions[mds_label][y_label_plot][m][1] for m in model_order if all_results_predictions[mds_label][y_label_plot][m][1] is not None]] if vals is not None])
                 if len(all_vals) > 0:
                     lims = [np.nanmin(all_vals), np.nanmax(all_vals)]
                     margin = (lims[1] - lims[0]) * 0.05 
                     lims = [lims[0] - margin, lims[1] + margin]
                     ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label='Ideal')
                     ax.set_xlim(lims)
                     ax.set_ylim(lims)
             
             ax.set_aspect('equal')
             ax.set_title(y_label_plot, fontsize=14)
             ax.set_xlabel("Actual Values", fontsize=12)
             if i == 0: ax.set_ylabel("Predicted Values", fontsize=12)
             ax.legend(fontsize=10)
             ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.savefig(os.path.join(output_dir, f"Pred_vs_Actual_{mds_label}.png"), dpi=300)
        plt.close() 

        # Display and Save Summary Table
        print(f"\n=== Summary Table: {mds_label} Subscore ===\n")
        df_subscore_display = df_subscore[['Model', 'Time Point', 'R2_test', 'RMSE_test', 'MAE_test', 'AIC_train', 'N_obs_test']]
        print(df_subscore_display.to_string(index=False)) # Print table to console
        
        excel_filename = os.path.join(output_dir, f"{mds_label}_model_summary.xlsx")
        try:
            df_subscore_display.to_excel(excel_filename, index=False)
            print(f"Saved model summary table to {excel_filename}")
        except Exception as e:
            print(f"Error saving Excel summary table {excel_filename}: {e}")
            
        try: # Optional image saving
            table_image_filename = os.path.join(output_dir, f"{mds_label}_model_summary.png")
            # dfi.export(df_subscore_display.style.format(precision=3), table_image_filename) # Requires Chrome/webdriver
            # print(f"Saved model summary table image to {table_image_filename}")
            pass # Skip image export for now
        except Exception as e:
            print(f"Error saving model summary table image {table_image_filename}: {e}")

    return df_summary, all_results_predictions

# --- Run the Analysis ---
if not train_df.empty and not test_df.empty:
    mdsupdrs_predictors_exist = [p for p in mdsupdrs_predictors_raw if p in train_df.columns]
    delta_ledd_targets_exist = [t for t in delta_ledd_targets if t in train_df.columns]
    
    if mdsupdrs_predictors_exist and delta_ledd_targets_exist:
        analysis_summary_df, analysis_predictions = perform_subscore_analysis(
            train_data=train_df,
            test_data=test_df,
            mdsupdrs_vars=mdsupdrs_predictors_exist,
            y_vars=delta_ledd_targets_exist,
            basic_covariates=final_basic_covariates,
            wb_fa_covariates=final_wb_fa_covariates,
            stable_features_per_model=stable_features_per_model, 
            tract_features=tract_feature_cols,
            output_dir=FIGURES_DIR 
        )
    else:
        print("Skipping OLS analysis due to missing predictor or target columns in data.")
        analysis_summary_df = pd.DataFrame()
        analysis_predictions = {}
else:
    print("Skipping OLS analysis due to empty train/test data.")
    analysis_summary_df = pd.DataFrame()
    analysis_predictions = {}

# Section V: Brain Visualization
# --- Brain Visualization Function ---
# (Code remains the same as before)
def plot_top_tracts_single_row(top_tract_names, top_tract_coefficients, model_id, output_dir):
    """
    Plots the top tracts based on LASSO coefficients in a single row (sagittal, coronal, axial).
    Uses 'coolwarm' colormap. model_id is like 'delta_ledd1_pred_MDSUPDRSIIIpre_Percent_TOTAL_V'
    """
    print(f"\n--- Plotting Brain Map for: {model_id} ---")
    
    tract_mapping = {
        'ATR L': 1, 'ATR R': 2, 'CST L': 3, 'CST R': 4, 'CGC L': 5, 'CGC R': 6, 
        'CGH L': 7, 'CGH R': 8, 'FMA': 9, 'FMI': 10, 'IFOF L': 11, 'IFOF R': 12, 
        'ILF L': 13, 'ILF R': 14, 'SLF L': 15, 'SLF R': 16, 'SLFT L': 17, 'SLFT R': 18, 
        'UF L': 19, 'UF R': 20
    }

    try:
        atlas_img = nib.load(ATLAS_FILE)
        template = datasets.load_mni152_template()
    except FileNotFoundError as e:
        print(f"Error loading atlas or template: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading atlas/template: {e}")
        return

    top_tracts_simplified = [simplify_feature_name(name) for name in top_tract_names]
    valid_indices = [i for i, tract in enumerate(top_tracts_simplified) if tract in tract_mapping]
    if not valid_indices:
        print(f"No valid tracts found in mapping for {model_id}. Skipping plot.")
        return
        
    top_tracts_simplified = [top_tracts_simplified[i] for i in valid_indices]
    top_coefficients_filtered = [top_tract_coefficients[i] for i in valid_indices]
    
    if not top_tracts_simplified: 
         print(f"No tracts left after mapping for {model_id}. Skipping plot.")
         return

    tract_label_mapping = {tract: i + 1 for i, tract in enumerate(top_tracts_simplified)}
    combined_tract_data = np.zeros(atlas_img.shape, dtype=np.int16)
    for tract, label in tract_label_mapping.items():
        intensity = tract_mapping[tract]
        tract_mask = (atlas_img.get_fdata() == intensity)
        combined_tract_data[tract_mask] = label
        
    combined_tract_img = nib.Nifti1Image(combined_tract_data, atlas_img.affine)
    resampled_tract_img = image.resample_to_img(combined_tract_img, template, interpolation='nearest')

    min_coeff = np.min(top_coefficients_filtered)
    max_coeff = np.max(top_coefficients_filtered)
    if min_coeff == max_coeff: vmin, vmax = min_coeff - 0.1, max_coeff + 0.1
    else: vmin, vmax = min_coeff, max_coeff
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_coeffs = cm.get_cmap('coolwarm') 
    
    label_to_color = {tract_label_mapping[tract]: cmap_coeffs(norm(coeff)) 
                      for tract, coeff in zip(top_tracts_simplified, top_coefficients_filtered)}
    plot_colors = [label_to_color.get(i, (0,0,0,0)) for i in range(1, len(top_tracts_simplified) + 1)] 
    custom_cmap_roi = ListedColormap(plot_colors)
    
    cut_coords = {'x': 0, 'y': -10, 'z': 10} 
    display_modes = ['x', 'y', 'z'] 

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    for i, display_mode in enumerate(display_modes):
        ax = axes[i]
        plotting.plot_roi(
            resampled_tract_img, bg_img=template, display_mode=display_mode,
            cut_coords=[cut_coords[display_mode]], axes=ax, cmap=custom_cmap_roi,
            colorbar=False, threshold=0.5, black_bg=False, annotate=False, alpha=0.8
        )
        ax.set_title(f"{display_mode.upper()}-view", fontsize=14) 
    plt.tight_layout()
    
    output_image_path = os.path.join(output_dir, f'{model_id}_tract_visualization_coolwarm.png')
    plt.savefig(output_image_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig) 
    print(f"Saved brain map: {output_image_path}")

    fig_colorbar, ax_colorbar = plt.subplots(figsize=(1.5, 6))
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_coeffs), cax=ax_colorbar, orientation='vertical')
    cbar.set_label('LASSO Coefficient', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    colorbar_image_path = os.path.join(output_dir, f'{model_id}_colorbar_coolwarm.png')
    plt.savefig(colorbar_image_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig_colorbar) 
    print(f"Saved colorbar: {colorbar_image_path}")

# --- Generate Brain Maps for each model ---
if stable_coeffs_dfs_per_model:
     for mds_var, target_dict in stable_coeffs_dfs_per_model.items():
          for target, df_coeffs in target_dict.items():
               # Filter for tract features only for visualization
               tract_coeffs_df = df_coeffs[df_coeffs['Feature'].isin(tract_feature_cols)]
               if not tract_coeffs_df.empty:
                    model_identifier = f"{target}_pred_{mds_var}" # Unique ID for file naming
                    # Get coefficients corresponding to the tract names
                    coeffs_for_plot = tract_coeffs_df['Mean_Coefficient'].tolist()
                    names_for_plot = tract_coeffs_df['Feature'].tolist()
                    
                    plot_top_tracts_single_row(
                         top_tract_names=names_for_plot, 
                         top_tract_coefficients=coeffs_for_plot, 
                         model_id=model_identifier,
                         output_dir=FIGURES_DIR
                    )
               else:
                    print(f"No stable tract features found for {target} (Predictor: {mds_var}) to visualize.")
else:
     print("Skipping brain visualization as no stable features were identified.")

print("\nAnalysis Script Complete.")
