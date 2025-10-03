import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
from scipy.stats import randint, uniform, loguniform

import lightgbm as lgb
import warnings

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
    print(" SHAP library available")
except ImportError:
    SHAP_AVAILABLE = False
    print(" SHAP library not available. Install with: pip install shap")

warnings.filterwarnings('ignore')

plt.ioff() # Turn off Matplotlib interactive mode
try:
    os.environ['OMP_NUM_THREADS'] = '4'
except Exception:
    pass

script_version = "Model_VFinal_RawData_SeparateDatasets" # Updated script version
print(f" Starting ML Pipeline {script_version}: Using Raw Data (No Feature Engineering)")
print(" Hyperparameters adjusted for small dataset and imbalanced classes")
print(" SCENARIO: Raw Sensor Data as Features - Separate Train/Test Files")

# === CONFIGURATION ===
SKIP_LIGHTGBM = False  # Set to True if LightGBM is too slow
INTENSIVE_SEARCH = True  # Set to False for faster execution
cv_folds_search = 10     # Using 10 folds for robust CV (as discussed)
APPLY_SHAP_ANALYSIS = True  # Enable SHAP analysis
SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP (to control computation time)

# Iterations based on previous discussion (higher for complex models)
n_iter_map = {
    'KNN': 60,
    'SVM': 100,
    'MLP': 300,
    'LightGBM': 200
}

# If INTENSIVE_SEARCH is False, reduce iterations further
if not INTENSIVE_SEARCH:
    n_iter_map = {'KNN': 50, 'SVM': 50, 'MLP': 50, 'LightGBM': 10}

print(f" Configuration: SKIP_LIGHTGBM={SKIP_LIGHTGBM}, INTENSIVE_SEARCH={INTENSIVE_SEARCH}")
print(f" CV Folds: {cv_folds_search}, Iterations: {n_iter_map}")
print(f" SHAP Analysis: {APPLY_SHAP_ANALYSIS and SHAP_AVAILABLE}")

output_folder = f'ML_Results_RawData_Separate_{script_version}' # Updated output folder
os.makedirs(output_folder, exist_ok=True)
plots_folder = os.path.join(output_folder, 'Plots')
os.makedirs(plots_folder, exist_ok=True)
shap_folder = os.path.join(output_folder, 'SHAP_Analysis')
os.makedirs(shap_folder, exist_ok=True)

# === Load datasets ===
# UPDATED PATHS HERE
DATASET_BASE_PATH = r'/home/strokmatic/Robot_Features/Preprocessed_Data'
DATASET_TRAIN_PATH = os.path.join(DATASET_BASE_PATH, 'dataset_train_original.csv')
DATASET_TEST_PATH = os.path.join(DATASET_BASE_PATH, 'dataset_test_original.csv')

try:
    df_train_raw = pd.read_csv(DATASET_TRAIN_PATH)
    print(f"\n Loaded training dataset: {DATASET_TRAIN_PATH} with shape {df_train_raw.shape}")
    print(f"NaN values in original training dataset: {df_train_raw.isna().sum().sum()}")
    print("Class distribution in training data:")
    print(df_train_raw['label'].value_counts())
except Exception as e:
    print(f" Error loading training dataset from {DATASET_TRAIN_PATH}: {e}")
    exit()

try:
    df_test_raw = pd.read_csv(DATASET_TEST_PATH)
    print(f"\n Loaded test dataset: {DATASET_TEST_PATH} with shape {df_test_raw.shape}")
    print(f"NaN values in original test dataset: {df_test_raw.isna().sum().sum()}")
    print("Class distribution in test data:")
    print(df_test_raw['label'].value_counts())
except Exception as e:
    print(f" Error loading test dataset from {DATASET_TEST_PATH}: {e}")
    exit()

variables = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# Ensure that variable columns are treated as lists of numbers for both train and test
for var in variables:
    df_train_raw[var] = df_train_raw[var].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_test_raw[var] = df_test_raw[var].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

print(f"\nNaN values after initial transformation (checking lists) in TRAIN: {df_train_raw[variables].isna().sum().sum()}")
print(f"NaN values after initial transformation (checking lists) in TEST: {df_test_raw[variables].isna().sum().sum()}")

# === Prepare Raw Data as Features ===
# This function will flatten the list-like features into individual columns
def flatten_raw_data(df, variables, max_len=None, is_training=True):
    current_max_len = 0
    if is_training:
        # Find the maximum length of the lists across all rows and variables in training data
        for var in variables:
            valid_lists = df[var].dropna()
            if not valid_lists.empty:
                current_max_len_var = valid_lists.apply(len).max()
                if current_max_len_var > current_max_len:
                    current_max_len = int(current_max_len_var)
        
        if current_max_len == 0:
            print("All feature lists are empty or NaN in training data. Cannot flatten.")
            return pd.DataFrame(), 0 # Return empty DataFrame and 0 max_len
        max_len_to_use = current_max_len
        print(f"Flattening raw data. Max length found in training data: {max_len_to_use} per variable.")
    else:
        # For testing data, use the max_len determined from the training data
        if max_len is None or max_len == 0:
            print(" max_len not provided for test data, or it is 0. Cannot flatten test data consistently.")
            return pd.DataFrame(), 0
        max_len_to_use = max_len
        print(f"Flattening test data using max length from training data: {max_len_to_use} per variable.")

    flattened_features = []
    for index, row in df.iterrows():
        row_features = {}
        for var in variables:
            values = np.array(row[var], dtype=float)
            
            # Pad with NaN if the list is shorter than max_len_to_use
            # Or handle empty/all NaN lists by filling with NaN for the whole sequence
            if values.size == 0 or np.all(np.isnan(values)):
                padded_values = np.full(max_len_to_use, np.nan)
            else:
                padded_values = np.pad(values, (0, max_len_to_use - values.size), 'constant', constant_values=np.nan)
            
            for i, val in enumerate(padded_values):
                row_features[f'{var}_{i+1}'] = val
        flattened_features.append(row_features)
    
    df_flattened = pd.DataFrame(flattened_features, index=df.index)
    
    # Handle NaNs introduced by padding or missing data: fill with median
    df_flattened = df_flattened.replace([np.inf, -np.inf], np.nan)
    if df_flattened.isnull().values.any():
        # Calculate median only for numeric columns
        median_values = df_flattened.median(numeric_only=True)
        if median_values.isnull().any():
            print("Some median values are NaN (e.g., column was all NaN). Filling those with 0.")
            median_values = median_values.fillna(0) # Fallback for columns entirely NaN
        
        print(f"NaNs found in flattened features. Filling with median (or 0 if median was NaN)...")
        df_flattened = df_flattened.fillna(median_values)
    
    # After filling, check if any column is still all NaN (e.g., if a variable was always empty/nan)
    if df_flattened.isnull().values.any():
        print("Some columns still contain NaNs after median fill. Filling remaining with 0.")
        df_flattened = df_flattened.fillna(0) # Fallback to 0 if median fails

    return df_flattened, max_len_to_use

# === SHAP Analysis Functions ===
def create_variable_mapping(feature_names, variables):
    variable_mapping = {}
    for feature in feature_names:
        for var in variables:
            if feature.startswith(f'{var}_'):
                variable_mapping[feature] = var
                break
    return variable_mapping

def aggregate_shap_by_variable(shap_values, feature_names, variables):
    """
    Aggregate SHAP values by main variables (Fx, Fy, Fz, Tx, Ty, Tz)
    """
    variable_mapping = create_variable_mapping(feature_names, variables)
    
    # Initialize aggregated SHAP values
    if len(shap_values.shape) == 3:  # Multi-class
        n_samples, n_features, n_classes = shap_values.shape
        aggregated_shap = np.zeros((n_samples, len(variables), n_classes))
        
        for i, var in enumerate(variables):
            var_features = [j for j, feat in enumerate(feature_names) if variable_mapping.get(feat) == var]
            if var_features:
                aggregated_shap[:, i, :] = np.sum(shap_values[:, var_features, :], axis=1)
                
    else:  # Binary classification or regression
        n_samples, n_features = shap_values.shape
        aggregated_shap = np.zeros((n_samples, len(variables)))
        
        for i, var in enumerate(variables):
            var_features = [j for j, feat in enumerate(feature_names) if variable_mapping.get(feat) == var]
            if var_features:
                aggregated_shap[:, i] = np.sum(shap_values[:, var_features], axis=1)
    
    return aggregated_shap

def calculate_shap_importance(model, X_sample, model_name, feature_names, variables):
    print(f"Calculating SHAP values for {model_name}...")
    
    try:
        # Choose appropriate explainer based on model type
        if model_name == 'LightGBM':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
        elif model_name in ['KNN', 'SVM', 'MLP']:
            # Explainer
            background_size = min(50, len(X_sample))
            background = shap.utils.sample(X_sample, background_size)
            explainer = shap.Explainer(model.predict, background)
            shap_values = explainer(X_sample)
            
            # Extract values from Explanation object
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values
        else:
            print(f" Unknown model type for SHAP: {model_name}")
            return None, None
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # Multi-class: shap_values is a list of arrays for each class
            shap_values = np.stack(shap_values, axis=-1)
        
        # Aggregate by main variables
        aggregated_shap = aggregate_shap_by_variable(shap_values, feature_names, variables)
        
        # Calculate mean absolute SHAP values for ranking
        if len(aggregated_shap.shape) == 3:  # Multi-class
            mean_abs_shap = np.mean(np.abs(aggregated_shap), axis=(0, 2))
        else:  # Binary or regression
            mean_abs_shap = np.mean(np.abs(aggregated_shap), axis=0)
        
        # Create ranking
        variable_importance = dict(zip(variables, mean_abs_shap))
        variable_ranking = sorted(variable_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"          SHAP analysis completed for {model_name}")
        return aggregated_shap, variable_ranking
        
    except Exception as e:
        print(f"          Error calculating SHAP for {model_name}: {e}")
        return None, None

def plot_shap_importance(variable_ranking, model_name, scenario_name):
    """
    Plot SHAP-based variable importance
    """
    if not variable_ranking:
        return
    
    variables_ranked, importance_values = zip(*variable_ranking)
    
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    bars = plt.bar(variables_ranked, importance_values, color=colors[:len(variables_ranked)])
    
    plt.title(f'SHAP Variable Importance - {model_name}\n{scenario_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Variables', fontsize=12)
    plt.ylabel('Mean |SHAP Value|', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, importance_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importance_values) * 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Add ranking text
    ranking_text = "Ranking:\n" + "\n".join([f"{i+1}. {var} ({val:.4f})" 
                                            for i, (var, val) in enumerate(variable_ranking)])
    plt.text(0.02, 0.98, ranking_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    
    filename = f'SHAP_Importance_{model_name}_{scenario_name.replace(" ", "_").replace("/", "_")}.png'
    filepath = os.path.join(shap_folder, filename)
    
    try:
        plt.savefig(filepath, dpi=180, bbox_inches='tight')
        print(f"   SHAP importance plot saved: {filename}")
    except Exception as e:
        print(f"   Error saving SHAP plot: {e}")
    plt.close()

def plot_combined_shap_ranking(all_rankings, scenario_name):
    """
    Plot combined SHAP rankings for all models
    """
    if not all_rankings:
        return
    
    plt.figure(figsize=(14, 8))
    
    # Create subplot for each model
    n_models = len(all_rankings)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'SHAP Variable Importance Comparison - {scenario_name}', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    for idx, (model_name, ranking) in enumerate(all_rankings.items()):
        if idx >= 4 or not ranking:  # Max 4 models
            continue
            
        variables_ranked, importance_values = zip(*ranking)
        
        bars = axes[idx].bar(variables_ranked, importance_values, color=colors[:len(variables_ranked)])
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Variables')
        axes[idx].set_ylabel('Mean |SHAP Value|')
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, importance_values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importance_values) * 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_models, 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    filename = f'SHAP_Combined_Rankings_{scenario_name.replace(" ", "_").replace("/", "_")}.png'
    filepath = os.path.join(shap_folder, filename)
    
    try:
        plt.savefig(filepath, dpi=180, bbox_inches='tight')
        print(f"     Combined SHAP rankings saved: {filename}")
    except Exception as e:
        print(f"     Error saving combined SHAP plot: {e}")
    plt.close()

def save_shap_rankings(all_rankings, scenario_name):
    """
    Save SHAP rankings to CSV
    """
    if not all_rankings:
        return
    
    # Create comprehensive ranking DataFrame
    ranking_data = []
    
    for model_name, ranking in all_rankings.items():
        if ranking:
            for rank, (variable, importance) in enumerate(ranking, 1):
                ranking_data.append({
                    'Model': model_name,
                    'Variable': variable,
                    'Rank': rank,
                    'SHAP_Importance': importance,
                    'Scenario': scenario_name
                })
    
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        
        # Save detailed rankings
        filename = f'SHAP_Rankings_{scenario_name.replace(" ", "_").replace("/", "_")}.csv'
        filepath = os.path.join(shap_folder, filename)
        ranking_df.to_csv(filepath, index=False)
        
        # Create summary table (average ranking across models)
        summary_data = []
        for var in variables:
            var_ranks = ranking_df[ranking_df['Variable'] == var]['Rank'].values
            var_importance = ranking_df[ranking_df['Variable'] == var]['SHAP_Importance'].values
            
            summary_data.append({
                'Variable': var,
                'Avg_Rank': np.mean(var_ranks),
                'Avg_SHAP_Importance': np.mean(var_importance),
                'Std_Rank': np.std(var_ranks),
                'Std_SHAP_Importance': np.std(var_importance)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Avg_Rank')
        
        # Save summary
        summary_filename = f'SHAP_Summary_{scenario_name.replace(" ", "_").replace("/", "_")}.csv'
        summary_filepath = os.path.join(shap_folder, summary_filename)
        summary_df.to_csv(summary_filepath, index=False)
        
        print(f"     SHAP rankings saved: {filename}")
        print(f"     SHAP summary saved: {summary_filename}")
        
        # Print summary to console
        print("\n     SHAP VARIABLE IMPORTANCE SUMMARY:")
        print("     " + "="*50)
        for _, row in summary_df.iterrows():
            print(f"     {row['Variable']:4s} | Avg Rank: {row['Avg_Rank']:.1f} | Avg Importance: {row['Avg_SHAP_Importance']:.4f}")
        print("     " + "="*50)

def perform_shap_analysis(trained_models, X_test_processed, feature_names, variables, scenario_name):
    """
    Perform SHAP analysis for all trained models
    """
    if not SHAP_AVAILABLE:
        print("   SHAP not available, skipping analysis")
        return
    
    print(f"\n    Starting SHAP Analysis for {scenario_name}...")
    
    # Sample data for SHAP (to control computation time)
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test_processed))
    sample_indices = np.random.choice(len(X_test_processed), sample_size, replace=False)
    X_sample = X_test_processed.iloc[sample_indices]
    
    print(f"     Using {sample_size} samples for SHAP analysis")
    
    all_rankings = {}
    
    for model_name, model_assets in trained_models.items():
        if model_assets and 'model' in model_assets:
            model = model_assets['model']
            
            # Calculate SHAP values and rankings
            shap_values, variable_ranking = calculate_shap_importance(
                model, X_sample, model_name, feature_names, variables
            )
            
            if variable_ranking:
                all_rankings[model_name] = variable_ranking
                
                # Plot individual model importance
                plot_shap_importance(variable_ranking, model_name, scenario_name)
    
    if all_rankings:
        # Plot combined rankings
        plot_combined_shap_ranking(all_rankings, scenario_name)
        
        # Save rankings
        save_shap_rankings(all_rankings, scenario_name)
    
    print(f"SHAP Analysis completed for {scenario_name}")

print("\nFlattening raw sensor data for training and testing...")

# First, flatten training data to determine the sequence_length
X_train_raw_features, sequence_length = flatten_raw_data(df_train_raw, variables, is_training=True)
if X_train_raw_features.empty:
    print("Flattened training features DataFrame is empty. Exiting.")
    exit()

print(f"Flattened raw training data shape: {X_train_raw_features.shape}")

# Then, flatten test data using the determined sequence_length
X_test_raw_features, _ = flatten_raw_data(df_test_raw, variables, max_len=sequence_length, is_training=False)
if X_test_raw_features.empty:
    print("Flattened test features DataFrame is empty. Exiting.")
    exit()

print(f"Flattened raw test data shape: {X_test_raw_features.shape}")
print(f"Total number of raw features (dimensions): {X_train_raw_features.shape[1]}")

# === Target preparation ===
y_train = df_train_raw['label']
y_test = df_test_raw['label']

le = LabelEncoder()
# Fit LabelEncoder on the combined unique labels from both train and test to ensure all classes are known
# This prevents errors if a class only appears in the test set, though good practice is to have all in train
combined_labels = pd.concat([y_train, y_test]).unique()
le.fit(combined_labels)

y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)
class_names = list(le.classes_)
num_classes = len(class_names)
print(f"\nTarget: {num_classes} classes: {class_names}")
print(f"Encoded class distribution in TRAIN: {dict(zip(class_names, np.bincount(y_train_encoded)))}")
print(f"Encoded class distribution in TEST: {dict(zip(class_names, np.bincount(y_test_encoded)))}")

# === Models and Hyperparameter Optimization ===
def get_models_with_random_search(intensive_search=True, skip_lightgbm=False, cv_folds=10):
    print("\n🔧 Configuring RandomizedSearchCV...")
    param_distributions = {
        'KNN': {
            'n_neighbors': [1, 2, 3, 5, 7, 9],
            
            # weights personalizado com função custom
            'weights': ['uniform', 'distance'] + [
                lambda d: 1 / (d + 1e-6),  # Inverse distance
                lambda d: np.exp(-d),       # Exponential decay
            ],
            
            # Métricas específicas por tipo de feature
            'metric': ['euclidean', 'manhattan', 'cosine'],
            
            # p otimizado por experimentação
            'p': [1, 1.2, 1.5, 2, 2.5, 3],
            
            # algorithm baseado no tamanho do dataset
            'algorithm': ['kd_tree']
        },
        'SVM': {
            'C': uniform(0.1, 1000),
            'gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, 10)),
            'kernel': ['rbf'],
            'degree': randint(2, 6),
            'coef0': uniform(-2, 4),
            'class_weight': ['balanced', None]
        },
        'MLP': {
            'hidden_layer_sizes': [(50,), (100,), (150,), (50, 25), (100, 50), (150, 75), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd', 'lbfgs'],
            'alpha': uniform(0.00005, 0.01),
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': uniform(0.0005, 0.01),
            'max_iter': [1000, 1500],
            'beta_1': uniform(0.8, 0.19),
            'beta_2': uniform(0.9, 0.099),
            'batch_size': [64, 128]
        },
        'LightGBM': {
            'n_estimators': randint(500, 2000), # Mais árvores
            'learning_rate': loguniform(0.005, 0.1), # Learning rate menor
            'num_leaves': randint(5, 50), # Mais folhas
            'max_depth': randint(3, 7), # Profundidade um pouco maior
            'subsample': uniform(0.5, 0.5), # Subsampling de 0.5 a 1.0
            'colsample_bytree': uniform(0.5, 0.5), # Feature subsampling de 0.5 a 1.0
            'reg_alpha': loguniform(0.01, 1000.0), # Regularização L1 mais ampla
            'reg_lambda': loguniform(0.01, 1000.0), # Regularização L2 mais ampla
            'min_child_samples': randint(10, 50), # Faixa um pouco mais ampla
            'min_data_in_leaf': randint(15, 50), # Faixa um pouco mais ampla
            'class_weight': ['balanced'],
            'verbosity': [-1],
            'feature_fraction': uniform(0.5, 0.5), # Feature subsampling de 0.5 a 1.0
            'bagging_fraction': uniform(0.5, 0.5), # Bagging de 0.5 a 1.0
            'bagging_freq': randint(1, 10) # Frequência de bagging
        }
    }
    
    base_models = {
        'KNN': KNeighborsClassifier(n_jobs=1),
        'SVM': SVC(probability=True,random_state=42),
        'MLP': MLPClassifier(random_state=42,early_stopping=True,n_iter_no_change=15, max_iter=1000),
    }
    
    if not skip_lightgbm:
        base_models['LightGBM'] = lgb.LGBMClassifier(
            random_state=42, 
            verbosity=-1, 
            n_jobs=1,
            force_row_wise=True,
            class_weight='balanced'
        )
    else:
        print("     ⚠️ Skipping LightGBM (skip_lightgbm=True)")

    optimized_models = {}
    for name, base_model in base_models.items():
        if name not in param_distributions or not param_distributions[name]:
            print(f"     ⚠️ {name} has no defined/valid parameter distributions. Using base model.")
            optimized_models[name] = clone(base_model)
            continue

        optimized_models[name] = RandomizedSearchCV(estimator=base_model,
                                                   param_distributions=param_distributions[name],
                                                   n_iter=n_iter_map.get(name, 100),
                                                   cv=StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=42),
                                                   scoring='f1_macro',
                                                   n_jobs=-1,
                                                   random_state=42,
                                                   verbose=0)
        print(f"    {name} (RandomizedSearchCV) configured ({n_iter_map.get(name, 100)} iters, {cv_folds}-Fold CV search, scoring='f1_macro')")
    return optimized_models

# === Training and Evaluation Function ===
def train_evaluate_models(X_tr_eval, y_tr_eval, X_te_eval, y_te_eval, scenario_name_eval, class_labels_eval, num_cls_eval_arg, cv_folds):
    print(f"\n Training/Evaluating for {scenario_name_eval}")
    if X_tr_eval.empty or X_te_eval.empty:
        print(f"     ⚠️ Empty training or test data for {scenario_name_eval}. Skipping.")
        return {},{},{}

    feature_names = X_tr_eval.columns.tolist() if isinstance(X_tr_eval, pd.DataFrame) else None
    
    # === Feature Standardization (ONLY on TRAIN set before search, then apply to test) ===
    print(f"      Standardizing data for {scenario_name_eval}...")
    scaler = StandardScaler()
    X_tr_processed_scaled = pd.DataFrame(scaler.fit_transform(X_tr_eval), columns=feature_names, index=X_tr_eval.index)
    X_te_processed_scaled = pd.DataFrame(scaler.transform(X_te_eval), columns=feature_names, index=X_te_eval.index)
    print(f"      Standardization completed.")

    print(f"     Train Dimensions: {X_tr_processed_scaled.shape}, Labels: {y_tr_eval.shape}")
    unique_labels_train, counts_train = np.unique(y_tr_eval, return_counts=True)
    print(f"     Class Distribution in Train Set: {dict(zip(le.inverse_transform(unique_labels_train), counts_train))}")

    models_to_tune_eval = get_models_with_random_search(
        intensive_search=INTENSIVE_SEARCH, skip_lightgbm=SKIP_LIGHTGBM, cv_folds=cv_folds
    )
    eval_results_dict, final_predictions_dict, trained_model_assets_dict = {}, {}, {}

    for model_name_eval_loop, search_cv_loop in models_to_tune_eval.items():
        print(f"\n   {model_name_eval_loop}...")
        
        try:
            search_cv_loop.fit(X_tr_processed_scaled, y_tr_eval)
            best_model_eval = search_cv_loop.best_estimator_
            best_params_cv = search_cv_loop.best_params_
            best_score_cv = search_cv_loop.best_score_
        except Exception as e_fit:
            print(f"    Error fitting model {model_name_eval_loop}: {e_fit}")
            import traceback; traceback.print_exc()
            eval_results_dict[model_name_eval_loop]=None; final_predictions_dict[model_name_eval_loop]=None; trained_model_assets_dict[model_name_eval_loop]=None
            continue

        if best_model_eval is None:
            print(f"    best_model_eval is None for {model_name_eval_loop} after fit attempt. Skipping.")
            continue

        y_pred_on_test_eval = best_model_eval.predict(X_te_processed_scaled)
        trained_model_assets_dict[model_name_eval_loop] = {
            'model': best_model_eval,
            'scaler': scaler,
            'best_params': best_params_cv,
            'best_cv_search_score': best_score_cv,
            'feature_names': feature_names,
            'X_test_processed': X_te_processed_scaled  # Store for SHAP analysis
        }
        final_predictions_dict[model_name_eval_loop] = {'y_true': y_te_eval, 'y_pred': y_pred_on_test_eval}
        
        acc_test = accuracy_score(y_te_eval,y_pred_on_test_eval)
        f1_w = f1_score(y_te_eval,y_pred_on_test_eval,average='weighted',zero_division=0)
        f1_m = f1_score(y_te_eval,y_pred_on_test_eval,average='macro',zero_division=0)
        precision_w = precision_score(y_te_eval, y_pred_on_test_eval, average='weighted', zero_division=0)
        recall_w = recall_score(y_te_eval, y_pred_on_test_eval, average='weighted', zero_division=0)
        precision_m = precision_score(y_te_eval, y_pred_on_test_eval, average='macro', zero_division=0)
        recall_m = recall_score(y_te_eval, y_pred_on_test_eval, average='macro', zero_division=0)
        
        eval_results_dict[model_name_eval_loop] = {
            'accuracy_test':acc_test,
            'f1_weighted_test':f1_w, 'f1_macro_test':f1_m,
            'precision_weighted_test': precision_w, 'recall_weighted_test': recall_w,
            'precision_macro_test': precision_m, 'recall_macro_test': recall_m,
            'best_params':best_params_cv,'best_cv_search_score':best_score_cv,
            'classification_report_test':classification_report(y_te_eval,y_pred_on_test_eval,target_names=class_labels_eval,zero_division=0, output_dict=False)
        }
        print(f" Test Accuracy: {acc_test:.4f} | F1 Weighted: {f1_w:.4f} | F1 Macro: {f1_m:.4f} (CV Score: {best_score_cv:.4f})")
    
    return eval_results_dict,final_predictions_dict,trained_model_assets_dict

# === Confusion Matrix Plotting Function ===
def plot_confusion_matrix_eval(y_true_cm, y_pred_cm, model_name_cm, scenario_name_cm, class_labels_cm):
    if y_true_cm is None or y_pred_cm is None or len(y_true_cm) == 0 or len(y_pred_cm) == 0:
        print(f"     ⚠️ Invalid data for plotting Confusion Matrix for {model_name_cm} in {scenario_name_cm}. Skipping.")
        return
    plt.figure(figsize=(max(8, len(class_labels_cm) * 1.8), max(6, len(class_labels_cm) * 1.3)))
    cm = confusion_matrix(y_true_cm, y_pred_cm)
    cm_sum_axis1 = cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.zeros_like(cm, dtype=float)
    np.divide(cm.astype('float'), cm_sum_axis1, out=cm_percent, where=cm_sum_axis1!=0)
    cm_percent *= 100
    annot_data = [[f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    sns.heatmap(cm, annot=annot_data, fmt='s', cmap='Blues',
                xticklabels=class_labels_cm, yticklabels=class_labels_cm,
                cbar_kws={'label': 'No. of Samples'})
    plt.title(f'Confusion Matrix - {model_name_cm}\n{scenario_name_cm}', fontsize=13)
    plt.xlabel('Predicted', fontsize=11); plt.ylabel('Actual', fontsize=11)
    plt.xticks(fontsize=9, rotation=45, ha="right"); plt.yticks(fontsize=9, rotation=0)
    
    acc_val = accuracy_score(y_true_cm, y_pred_cm)
    f1_macro_val = f1_score(y_true_cm, y_pred_cm, average='macro', zero_division=0)
    precision_macro_val = precision_score(y_true_cm, y_pred_cm, average='macro', zero_division=0)
    recall_macro_val = recall_score(y_true_cm, y_pred_cm, average='macro', zero_division=0)
    
    plt.figtext(0.02, 0.01, f'Accuracy: {acc_val:.4f}\nF1-Macro: {f1_macro_val:.4f}\nPrecision-Macro: {precision_macro_val:.4f}\nRecall-Macro: {recall_macro_val:.4f}',
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", alpha=0.7))
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    filename = f'CM_{model_name_cm}_{scenario_name_cm.replace(" ","_").replace("/","_")}.png'
    filepath = os.path.join(plots_folder,filename)
    try:
        plt.savefig(filepath,dpi=180,bbox_inches='tight')
        print(f"   Confusion matrix saved: {filename}")
    except Exception as e_save:
        print(f"   Error saving Confusion Matrix '{filename}': {e_save}")
    plt.close()

# === SCENARIO EXECUTION (Raw Data) ===
results_raw, preds_raw, models_raw = {}, {}, {}

if not X_train_raw_features.empty and not X_test_raw_features.empty:
    results_raw, preds_raw, models_raw = train_evaluate_models(
        X_train_raw_features, y_train_encoded, X_test_raw_features, y_test_encoded,
        "Scenario_Raw_Data", class_names, num_classes, cv_folds_search
    )
    
    # Perform SHAP Analysis
    if APPLY_SHAP_ANALYSIS and models_raw:
        # Get processed test data from any model (they all use the same scaler)
        first_model_assets = next(iter(models_raw.values()))
        if first_model_assets and 'X_test_processed' in first_model_assets:
            X_test_processed = first_model_assets['X_test_processed']
            feature_names = first_model_assets['feature_names']
            
            perform_shap_analysis(models_raw, X_test_processed, feature_names, variables, "Raw_Data_Scenario")
    
    if preds_raw:
        print("\n  Generating confusion matrices - Raw Data Scenario...")
        for name, pred_data in preds_raw.items():
            if pred_data and 'y_true' in pred_data and 'y_pred' in pred_data:
                plot_confusion_matrix_eval(pred_data['y_true'], pred_data['y_pred'], name, "Scenario_Raw_Data_Separate", class_names)
else:
    print("⚠️ Raw Data Scenario cannot be executed (training or test data is empty).")

print("\n" + "="*70 + f"\n FINAL REPORT - {output_folder}\n" + "="*70)
comparison_data_final = []
scenario_lbl = "Raw Data Features (Separate Train/Test)" # Updated scenario label
results_dict_loop = results_raw

if not results_dict_loop:
    comparison_data_final.append({
        'Scenario':scenario_lbl,'Model':"N/A",
        'Test Accuracy':"N/A",'Test F1 (Weighted)':"N/A", 'Test F1 (Macro)':"N/A",
        'Test Precision (W)':"N/A", 'Test Recall (W)':"N/A",
        'Test Precision (M)':"N/A", 'Test Recall (M)':"N/A",
        'CV Search F1-Macro':"N/A"
    });
else:
    for model_nm, res_data in results_dict_loop.items():
        if res_data:
            comparison_data_final.append({
                'Scenario':scenario_lbl,'Model':model_nm,
                'Test Accuracy':f"{res_data.get('accuracy_test',0.0):.4f}",
                'Test F1 (Weighted)':f"{res_data.get('f1_weighted_test',0.0):.4f}",
                'Test F1 (Macro)':f"{res_data.get('f1_macro_test',0.0):.4f}",
                'Test Precision (W)':f"{res_data.get('precision_weighted_test',0.0):.4f}",
                'Test Recall (W)':f"{res_data.get('recall_weighted_test',0.0):.4f}",
                'Test Precision (M)':f"{res_data.get('precision_macro_test',0.0):.4f}",
                'Test Recall (M)':f"{res_data.get('recall_macro_test',0.0):.4f}",
                'CV Search F1-Macro':f"{res_data.get('best_cv_search_score',0.0):.4f}"
            })
        else:
            comparison_data_final.append({
                'Scenario':scenario_lbl,'Model':model_nm,
                'Test Accuracy':"Failed",'Test F1 (Weighted)':"Failed", 'Test F1 (Macro)':"Failed",
                'Test Precision (W)':"Failed", 'Test Recall (W)':"Failed",
                'Test Precision (M)':"Failed", 'Test Recall (M)':"Failed",
                'CV Search F1-Macro':"Failed"
            })

if comparison_data_final:
    comparison_df = pd.DataFrame(comparison_data_final)
    print("\n Performance Comparison:"); print(comparison_df.to_string(index=False))
    csv_p = os.path.join(output_folder,f'model_comparison_{os.path.basename(output_folder)}.csv'); comparison_df.to_csv(csv_p,index=False); print(f"\n     💾 Table saved: {csv_p}")
else:
    print("\n No comparison to display.")

print("\n BEST RESULT (Test F1-Macro):")
best_model_name, best_f1_macro = "N/A", -1.0
if results_raw: 
    valid_results = {k:v for k,v in results_raw.items() if v and 'f1_macro_test' in v}
    if valid_results:
        best_model_name = max(valid_results, key=lambda k:valid_results[k]['f1_macro_test'])
        best_f1_macro = valid_results[best_model_name]['f1_macro_test']
print(f"     Model: {best_model_name} (F1-Macro: {best_f1_macro:.4f})")

print(f"\n FINAL SUMMARY ({output_folder}):")
print(f"   Using Raw Data as Features (Flattened Sensor Sequences)")
print(f"   Training Dataset Size: {len(X_train_raw_features)} samples")
print(f"   Test Dataset Size: {len(X_test_raw_features)} samples")
print(f"   Total Raw Features: {X_train_raw_features.shape[1]}")
print(f"   Models: KNN, SVM, MLP, LightGBM")
print(f"   Cross-Validation: {cv_folds_search}-fold Stratified CV (on training data)")
print(f"   Hyperparameter Search: RandomizedSearchCV (Iterations: {n_iter_map})")
print(f"   Evaluation Metric: F1-Macro (for imbalanced classes)")
print(f"   Class Weighting: 'balanced' option included in search")
if APPLY_SHAP_ANALYSIS and SHAP_AVAILABLE:
    print(f"SHAP Analysis: Variable importance ranking completed")
    print(f"SHAP Results saved in: {shap_folder}")
print(f"\n PROCESS FINISHED! Results in: {os.path.abspath(output_folder)}")

