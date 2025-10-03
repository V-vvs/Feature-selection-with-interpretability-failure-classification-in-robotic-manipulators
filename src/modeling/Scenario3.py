import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import sys
import time

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
from scipy.stats import randint, uniform, loguniform

import lightgbm as lgb
import shap
import warnings

# --- PLOTLY IMPORTS ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings('ignore')

plt.ioff() # Turn off Matplotlib interactive mode
try:
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['PYTHONUNBUFFERED'] = '1'
except Exception:
    pass

# Configure Plotly renderer
pio.renderers.default = "png"

script_version = "ML_Pipeline_Scenario3_SHAP_Final_Enhanced_FocusedHyperspace"
print(f" Starting ML Pipeline {script_version}")
print(" SCENARIO 3: Filtered Feature Dataset from ReliefF_ANOVA_Correlation_Results/")
print(" SHAP analysis will be performed at the END")
print(" Enhanced with Precision/Recall Macro metrics (12 decimal precision)")
print(" Using Focused Hyperparameter Space (60-75% probability of optimal values)")

# === CONFIGURATION ===
SKIP_LIGHTGBM = False  # Set to True if LightGBM is too slow
INTENSIVE_SEARCH = True  # Set to False for faster execution

# HYPERPARAMETER MODES:
# True: Hyperparameter optimization with RandomizedSearchCV
# False: Fixed pre-optimized parameters  
# "default": Use scikit-learn default parameters
OPTIMIZE_HYPERPARAMETERS = "True"  

DATASET_PATH = r'./ReliefF_ANOVA_Correlation_Results'  # Path to filtered features data

print(f" Configuration: SKIP_LIGHTGBM={SKIP_LIGHTGBM}, INTENSIVE_SEARCH={INTENSIVE_SEARCH}")
if OPTIMIZE_HYPERPARAMETERS == False:
    print(f"⚙️ Hyperparameter Mode: OPTIMIZATION (RandomizedSearchCV)")
elif OPTIMIZE_HYPERPARAMETERS == True:
    print(f"⚙️ Hyperparameter Mode: FIXED PRE-OPTIMIZED PARAMETERS")
elif OPTIMIZE_HYPERPARAMETERS == "default":
    print(f"⚙️ Hyperparameter Mode: SCIKIT-LEARN DEFAULT PARAMETERS")
print(f" Dataset path: {DATASET_PATH}")

# === OUTPUT CONFIGURATION ===
output_folder = f'ML_Results_Scenario3_SHAP_Final_{script_version}'
os.makedirs(output_folder, exist_ok=True)
plots_folder = os.path.join(output_folder, 'Plots')
os.makedirs(plots_folder, exist_ok=True)
shap_folder = os.path.join(output_folder, 'SHAP_Analysis')
os.makedirs(shap_folder, exist_ok=True)

# === HYPERPARAMETER SPACE ===
def get_hyperparameter_space():
    """
    Hyperparameter space optimized for filtered feature dataset
    """
    if OPTIMIZE_HYPERPARAMETERS != True:
        return {}
    
    return {
        'KNN': {
            'n_neighbors': randint(3, 25),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'p': [1, 2]
        },
        'SVM': {
            'C': loguniform(0.01, 100),
            'gamma': loguniform(0.001, 1),
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'class_weight': ['balanced', None]
        },
        'MLP': {
            'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (150, 75), (200, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'lbfgs'],
            'alpha': loguniform(0.0001, 0.1),
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': loguniform(0.001, 0.1)
        },
        'LightGBM': {
            'n_estimators': randint(50, 500),
            'num_leaves': randint(10, 100),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': loguniform(0.01, 10.0),
            'reg_lambda': loguniform(0.01, 10.0),
            'min_child_samples': randint(5, 50),
            'learning_rate': loguniform(0.01, 0.2),
            'max_depth': randint(3, 10)
        }
    }

# === MODEL CONFIGURATION ===
def get_search_iterations(n_features, n_samples):
    """
    Configure search iterations based on dataset size
    Filtered features typically allow for more intensive search
    """
    if OPTIMIZE_HYPERPARAMETERS != True:
        return {
            'KNN': 0,
            'SVM': 0,
            'MLP': 0,
            'LightGBM': 0
        }
    
    feature_ratio = n_features / n_samples
    print(f" Dataset info: {n_samples} samples, {n_features} features")
    print(f" Feature/Sample ratio: {feature_ratio:.3f}")
    
    # More intensive search for filtered features (fewer features)
    iterations = {
        'KNN': 60,
        'SVM': 150,
        'MLP': 200,
        'LightGBM': 200
    }
    
    if feature_ratio < 0.2:
        print(" Low dimensionality - using intensive search")
        # Can afford more iterations with fewer features
        iterations = {
            'KNN': 80,
            'SVM': 200,
            'MLP': 200,
            'LightGBM': 200
        }
    elif feature_ratio < 0.4:
        print(" Balanced dimensionality - using standard intensive search")
    else:
        print(" High dimensionality detected - using conservative search")
    
    return iterations

def get_optimized_models(X_train, skip_lightgbm=False):
    """
    Create optimized models for filtered features scenario
    """
    print(f"\n Setting up models for filtered features scenario")
    if OPTIMIZE_HYPERPARAMETERS == True:
        print(f"⚙️ Mode: Hyperparameter optimization")
    elif OPTIMIZE_HYPERPARAMETERS == False:
        print(f"⚙️ Mode: Fixed pre-optimized parameters")
    elif OPTIMIZE_HYPERPARAMETERS == "default":
        print(f"⚙️ Mode: Scikit-learn default parameters")
    
    # === SCIKIT-LEARN DEFAULT PARAMETERS ===
    if OPTIMIZE_HYPERPARAMETERS == "default":
        default_models = {
            'KNN': KNeighborsClassifier(),  # All defaults: n_neighbors=5, weights='uniform', etc.
            'SVM': SVC(
                probability=True,  # Only change: enable probabilities for consistency
                random_state=42    # Add random_state for reproducibility
            ),
            'MLP': MLPClassifier(
                random_state=42,   # Add random_state for reproducibility
                max_iter=500       # Increase from 200 to avoid convergence warnings
            ),
            'LightGBM': lgb.LGBMClassifier(
                random_state=42,   # Add random_state for reproducibility
                verbose=-1         # Suppress warnings
            )
        }
        
        if skip_lightgbm:
            default_models.pop('LightGBM', None)
            print(" LightGBM skipped as requested")
        
        print(" Using scikit-learn default parameters")
        for name in default_models.keys():
            print(f"    {name}: Default parameters")
        
        return default_models
    
    # === FIXED PRE-OPTIMIZED PARAMETERS ===
    base_models = {
        'KNN': KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',
            algorithm='kd_tree',
            p=2
        ),
        'SVM': SVC(
            probability=True,
            random_state=42,
            C=10,
            gamma=0.015,
            tol=1,
            kernel='rbf',
            class_weight='balanced',
            
        ),
        'MLP': MLPClassifier(
            random_state=42, 
            early_stopping=True, 
            n_iter_no_change=20, 
            max_iter=2000, 
            hidden_layer_sizes=(75,100), 
            activation='relu',
            solver='lbfgs',
            alpha=0.05,
            learning_rate='invscaling',
            learning_rate_init=0.001,
            tol=0.005
        ),
        'LightGBM': lgb.LGBMClassifier(
            num_class=3,  
            metric='multi_logloss',  
            boosting_type='gbdt',
            verbose=-1,
            random_state=42,
            num_leaves=50,              
            max_depth=5,                
            min_data_in_leaf=55,        
            min_child_samples=10,       
            min_split_gain=0.02,        
            learning_rate=0.06,         
            n_estimators=1000,          
            #lambda_l1=0.01,             
            lambda_l2=0.05,             
            feature_fraction=1.0,       # 100% das features
            bagging_fraction=1,       # 100% das amostras
            bagging_freq=10,             # Frequência do bagging
            class_weight='balanced',    # Ajuste automático
            is_unbalance=True,          # Flag adicional
            force_col_wise=True,        # Otimização para dataset pequeno
            deterministic=True          # Reprodutibilidade
        )
    }
    
    if skip_lightgbm:
        base_models.pop('LightGBM', None)
        print("    LightGBM skipped as requested")
    
    # If using fixed parameters, return base models directly
    if OPTIMIZE_HYPERPARAMETERS == True:
        print("    Using fixed pre-optimized parameters")
        optimized_models = {}
        for name, model in base_models.items():
            optimized_models[name] = clone(model)
            print(f"    {name}: Fixed parameters (no optimization)")
        return optimized_models
    
    # If optimizing, set up RandomizedSearchCV
    print("    Setting up hyperparameter optimization")
    
    # Hyperparameter space
    param_distributions = get_hyperparameter_space()
    
    # Search iterations
    search_iterations = get_search_iterations(X_train.shape[1], X_train.shape[0])
    cv_folds = 10
    
    optimized_models = {}
    for name, base_model in base_models.items():
        if name not in param_distributions:
            optimized_models[name] = clone(base_model)
            print(f"   {name}: No hyperparameter space defined - using base model")
            continue
        
        optimized_models[name] = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions[name],
            n_iter=search_iterations[name],
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='f1_macro',
            n_jobs=2,
            random_state=42,
            verbose=1
        )
        
        print(f"   {name}: {search_iterations[name]} iterations, {cv_folds}-fold CV")
    
    return optimized_models

# === DATASET LOADING ===
def load_dataset():
    """
    Load filtered features dataset from ReliefF_ANOVA_Correlation_Results
    """
    print(f"\n Loading filtered features dataset...")
    
    train_file = os.path.join(DATASET_PATH, 'dataset_train_ReliefF_ANOVA_filtered.csv')
    test_file = os.path.join(DATASET_PATH, 'dataset_test_ReliefF_ANOVA_filtered.csv')
    
    # Check if files exist
    if not os.path.exists(train_file):
        print(f" Train file not found: {train_file}")
        print(f"   Please make sure ReliefF_ANOVA feature selection has been run first")
        return None, None, None, None, None
    
    if not os.path.exists(test_file):
        print(f" Test file not found: {test_file}")
        print(f"   Please make sure ReliefF_ANOVA feature selection has been run first")
        return None, None, None, None, None
    
    # Load datasets
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        print(f" Train dataset: {train_df.shape}")
        print(f" Test dataset: {test_df.shape}")
        
        # Check for label column
        if 'label' not in train_df.columns or 'label' not in test_df.columns:
            print(f" 'label' column not found in datasets")
            return None, None, None, None, None
        
        # Separate features and labels
        X_train = train_df.drop('label', axis=1)
        y_train_labels = train_df['label']
        X_test = test_df.drop('label', axis=1)
        y_test_labels = test_df['label']
        
        # Encode labels
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_labels)
        y_test = le.transform(y_test_labels)
        
        class_names = list(le.classes_)
        
        print(f" Filtered features: {X_train.shape[1]}")
        print(f" Classes: {class_names}")
        print(f" Train samples: {X_train.shape[0]}")
        print(f" Test samples: {X_test.shape[0]}")
        
        # Check class distribution
        unique_labels, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(le.inverse_transform(unique_labels), counts))
        print(f" Class distribution: {class_dist}")
        
        # Show some filtered features
        print(f"\n Sample of filtered features (first 10):")
        for i, feature in enumerate(X_train.columns[:10]):
            print(f"   {i+1:2d}. {feature}")
        if len(X_train.columns) > 10:
            print(f"   ... and {len(X_train.columns) - 10} more features")
        
        return X_train, X_test, y_train, y_test, le
        
    except Exception as e:
        print(f" Error loading datasets: {e}")
        return None, None, None, None, None

# === TRAINING AND EVALUATION ===
def train_evaluate_models(X_train, y_train, X_test, y_test, class_names):
    """
    Train and evaluate all models with enhanced metrics
    """
    print(f"\n Training and evaluating models...")
    if OPTIMIZE_HYPERPARAMETERS == True:
        print(f"⚙️ Mode: Hyperparameter Optimization")
    elif OPTIMIZE_HYPERPARAMETERS == False:
        print(f"⚙️ Mode: Fixed Parameters")
    elif OPTIMIZE_HYPERPARAMETERS == "default":
        print(f"⚙️ Mode: Default Parameters")
    
    feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
    
    # Get optimized models
    models = get_optimized_models(X_train, skip_lightgbm=SKIP_LIGHTGBM)
    
    results = {}
    predictions = {}
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\n    Training {model_name}...")
        sys.stdout.flush()
        
        if OPTIMIZE_HYPERPARAMETERS == True and isinstance(model, RandomizedSearchCV):
            print(f"       Hyperparameter optimization started (this may take a few minutes)...")
        elif OPTIMIZE_HYPERPARAMETERS == True:
            print(f"       Using fixed pre-optimized parameters")
        elif OPTIMIZE_HYPERPARAMETERS == "default":
            print(f"       Using scikit-learn default parameters")
        
        if model_name == 'LightGBM':
            print(f"       LightGBM training started...")
            sys.stdout.flush()
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame for consistency
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index) if feature_names else X_train_scaled
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index) if feature_names else X_test_scaled
        
        # Train model
        try:
            if OPTIMIZE_HYPERPARAMETERS == True and isinstance(model, RandomizedSearchCV):
                model.fit(X_train_scaled_df, y_train)
                best_model = model.best_estimator_
                best_params = model.best_params_
                best_cv_score = model.best_score_
            else:
                model.fit(X_train_scaled_df, y_train)
                best_model = model
                if OPTIMIZE_HYPERPARAMETERS == False:
                    best_params = "Fixed Parameters (No Optimization)"
                elif OPTIMIZE_HYPERPARAMETERS == "default":
                    best_params = "Scikit-Learn Default Parameters"
                else:
                    best_params = "N/A (Base Model)"
                y_train_pred = best_model.predict(X_train_scaled_df)
                best_cv_score = f1_score(y_train, y_train_pred, average='macro')
            
            # Make predictions
            y_pred = best_model.predict(X_test_scaled_df)
            
            # Store model assets
            trained_models[model_name] = {
                'model': best_model,
                'scaler': scaler,
                'best_params': best_params,
                'best_cv_score': best_cv_score,
                'feature_names': feature_names
            }
            
            # Store predictions
            predictions[model_name] = {
                'y_true': y_test,
                'y_pred': y_pred
            }
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            
            # Store results with enhanced metrics
            results[model_name] = {
                'accuracy_test': accuracy,
                'f1_weighted_test': f1_weighted,
                'f1_macro_test': f1_macro,
                'precision_weighted_test': precision_weighted,
                'recall_weighted_test': recall_weighted,
                'precision_macro_test': precision_macro,
                'recall_macro_test': recall_macro,
                'best_params': best_params,
                'best_cv_search_score': best_cv_score,
                'classification_report_test': classification_report(y_test, y_pred, target_names=class_names, zero_division=0, output_dict=False)
            }
            
            print(f"       Accuracy: {accuracy:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}")
            print(f"       CV Score: {best_cv_score:.4f}")
            print(f"       Precision-Macro: {precision_macro:.4f} | Recall-Macro: {recall_macro:.4f}")
            
        except Exception as e:
            print(f"       Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results, predictions, trained_models

# === SHAP ANALYSIS ===
def generate_shap_analysis(trained_models, X_test, X_train, class_names):
    """
    Generate SHAP analysis for all models using ALL available data
    """
    print(f"\n Starting SHAP Analysis")
    print(f"    Analyzing ALL {X_test.shape[0]} test samples")
    print(f"    Using ALL {X_train.shape[0]} training samples as background")
    
    if X_test.empty:
        print(f"   ⚠️ Test data is empty. Skipping SHAP analysis.")
        return {}
    
    feature_names = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else None
    shap_rankings = {}
    
    for model_name, model_assets in trained_models.items():
        if not model_assets or 'model' not in model_assets:
            continue
        
        print(f"\n    Calculating SHAP for {model_name}...")
        
        model = model_assets['model']
        scaler = model_assets['scaler']
        model_feature_names = model_assets.get('feature_names', feature_names)
        
        try:
            # Prepare data
            X_test_current = X_test
            X_train_current = X_train
            
            if isinstance(X_test, pd.DataFrame) and model_feature_names:
                if list(X_test.columns) != model_feature_names:
                    X_test_current = X_test[model_feature_names]
            
            if isinstance(X_train, pd.DataFrame) and model_feature_names:
                if list(X_train.columns) != model_feature_names:
                    X_train_current = X_train[model_feature_names]
            
            # Scale data
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_current), columns=model_feature_names)
            X_train_scaled = pd.DataFrame(scaler.transform(X_train_current), columns=model_feature_names)
            
            # Calculate SHAP values
            shap_values = None
            
            if isinstance(model, lgb.LGBMClassifier):
                print(f"      Using TreeExplainer for {model_name}")
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test_scaled)
                except Exception as e:
                    print(f"      TreeExplainer failed: {e}. Using KernelExplainer...")
                    explainer = shap.KernelExplainer(model.predict_proba, X_train_scaled)
                    shap_values = explainer.shap_values(X_test_scaled, nsamples='auto')
            else:
                print(f"      Using KernelExplainer for {model_name}")
                explainer = shap.KernelExplainer(model.predict_proba, X_train_scaled)
                shap_values = explainer.shap_values(X_test_scaled, nsamples='auto')
            
            if shap_values is None:
                print(f"       SHAP explainer returned None for {model_name}")
                continue
            
            # Process SHAP values
            num_classes = len(class_names)
            
            if isinstance(shap_values, list) and len(shap_values) == num_classes:
                shap_values_processed = shap_values
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 2 and num_classes == 2:
                    shap_values_processed = [-shap_values, shap_values]
                elif shap_values.ndim == 3:
                    if shap_values.shape[2] == num_classes:
                        shap_values_processed = [shap_values[:, :, i] for i in range(num_classes)]
                    elif shap_values.shape[0] == num_classes:
                        shap_values_processed = [shap_values[i, :, :] for i in range(num_classes)]
                    else:
                        print(f"       Unexpected SHAP array shape: {shap_values.shape}")
                        continue
                else:
                    print(f"       Unexpected SHAP format: {shap_values.shape}")
                    continue
            else:
                print(f"       Unrecognized SHAP format: {type(shap_values)}")
                continue
            
            # Calculate feature importance
            mean_abs_shaps_per_class = []
            for i, class_name in enumerate(class_names):
                if i < len(shap_values_processed):
                    class_shap_values = shap_values_processed[i]
                    mean_abs_shap = np.abs(class_shap_values).mean(axis=0)
                    mean_abs_shaps_per_class.append(mean_abs_shap)
                    print(f"         Class '{class_name}': calculated from {class_shap_values.shape[0]} samples")
            
            if not mean_abs_shaps_per_class:
                print(f"       No valid SHAP values for {model_name}")
                continue
            
            # Overall importance
            overall_importance = np.mean(mean_abs_shaps_per_class, axis=0)
            feature_importance = {
                (model_feature_names[j] if model_feature_names else f"Feature_{j}"): overall_importance[j] 
                for j in range(len(overall_importance))
            }
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            shap_rankings[model_name] = {
                'feature_ranking': sorted_features,
                'shap_values_per_class': {class_names[i]: mean_abs_shaps_per_class[i] for i in range(len(mean_abs_shaps_per_class))},
                'overall_importance': feature_importance
            }
            
            print(f"       SHAP analysis completed for {model_name}")
            print(f"         Top 5 features: {[f[0] for f in sorted_features[:5]]}")
            
            # Generate SHAP plots
            generate_shap_plots(shap_values_processed, X_test_current, model_feature_names, class_names, model_name, sorted_features)
            
        except Exception as e:
            print(f"       Error in SHAP calculation for {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return shap_rankings

def generate_shap_plots(shap_values_processed, X_test, feature_names, class_names, model_name, sorted_features):
    """
    Generate SHAP visualization plots
    """
    try:
        print(f"          Generating SHAP plots for {model_name}...")
        
        # Summary plot for each class
        for class_idx, class_name in enumerate(class_names):
            if class_idx < len(shap_values_processed):
                plt.figure(figsize=(12, max(6, len(feature_names) * 0.4) if feature_names else 8))
                
                try:
                    shap.summary_plot(
                        shap_values_processed[class_idx],
                        X_test,
                        feature_names=feature_names,
                        show=False,
                        max_display=20
                    )
                    plt.title(f'SHAP Summary - {model_name} - Class: {class_name}', fontsize=12)
                    
                    filename = f'SHAP_Summary_{model_name}_Class_{class_name}.png'
                    filepath = os.path.join(shap_folder, filename)
                    plt.savefig(filepath, dpi=150, bbox_inches='tight')
                    print(f"             SHAP summary plot saved: {filename}")
                    
                except Exception as e:
                    print(f"            ⚠️ Error creating SHAP summary for class {class_name}: {e}")
                
                plt.close()
        
        # Feature importance plot
        plt.figure(figsize=(12, max(8, len(sorted_features[:20]) * 0.4)))
        top_features = sorted_features[:20]
        feature_names_plot = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]
        
        y_pos = np.arange(len(feature_names_plot))
        plt.barh(y_pos, importance_values, color='lightcoral', alpha=0.8)
        plt.yticks(y_pos, feature_names_plot)
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Feature Importance (SHAP) - {model_name}')
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(importance_values):
            plt.text(v + max(importance_values) * 0.01, i, f'{v:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        filename = f'SHAP_Importance_{model_name}.png'
        filepath = os.path.join(shap_folder, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"             SHAP importance plot saved: {filename}")
        plt.close()
        
    except Exception as e:
        print(f"          Error generating SHAP plots for {model_name}: {e}")

# === CONFUSION MATRIX PLOTS ===
def plot_confusion_matrices(predictions, class_names):
    """
    Generate confusion matrix plots for all models
    """
    print(f"\n Generating confusion matrices...")
    
    n_models = len([name for name, pred_data in predictions.items() 
                    if pred_data and 'y_true' in pred_data and 'y_pred' in pred_data])
    
    if n_models == 0:
        print(f"   ⚠️ No valid predictions available")
        return
    
    # Determine layout
    if n_models == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    elif n_models == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    elif n_models <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    else:
        rows = (n_models + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
        axes = axes.flatten()
    
    fig.suptitle('Confusion Matrices - Filtered Features Scenario', fontsize=16, fontweight='bold')
    
    plot_idx = 0
    for model_name, pred_data in predictions.items():
        if not pred_data or 'y_true' not in pred_data or 'y_pred' not in pred_data:
            continue
        
        y_true = pred_data['y_true']
        y_pred = pred_data['y_pred']
        
        if len(y_true) == 0 or len(y_pred) == 0:
            continue
        
        ax = axes[plot_idx]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_sum_axis1 = cm.sum(axis=1)[:, np.newaxis]
        cm_percent = np.zeros_like(cm, dtype=float)
        np.divide(cm.astype('float'), cm_sum_axis1, out=cm_percent, where=cm_sum_axis1!=0)
        cm_percent *= 100
        
        # Create annotations
        annot_data = [[f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)"
                       for j in range(cm.shape[1])]
                       for i in range(cm.shape[0])]
        
        # Plot heatmap
        sns.heatmap(cm, annot=annot_data, fmt='s', cmap='Oranges',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar_kws={'label': 'No. of Samples'})
        
        # Calculate metrics
        acc_val = accuracy_score(y_true, y_pred)
        f1_macro_val = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted_val = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        ax.set_title(f'{model_name}\nAcc: {acc_val:.3f} | F1-M: {f1_macro_val:.3f} | F1-W: {f1_weighted_val:.3f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        
        plot_idx += 1
    
    # Remove extra subplots
    for i in range(plot_idx, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    
    # Save plot
    filename = 'confusion_matrices_filtered_features.png'
    filepath = os.path.join(plots_folder, filename)
    
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    Confusion matrices saved: {filename}")
    except Exception as e:
        print(f"    Error saving confusion matrices: {e}")
    
    plt.close()

# === RESULTS VISUALIZATION ===
def plot_model_comparison(results):
    """
    Create comparison plots of model performance
    """
    print(f"\n Generating model comparison plots...")
    
    if not results:
        print(f"   ⚠️ No results available for plotting")
        return
    
    # Prepare data
    models = []
    accuracies = []
    f1_macros = []
    f1_weighteds = []
    precisions = []
    recalls = []
    
    for model_name, result in results.items():
        if result:
            models.append(model_name)
            accuracies.append(result.get('accuracy_test', 0))
            f1_macros.append(result.get('f1_macro_test', 0))
            f1_weighteds.append(result.get('f1_weighted_test', 0))
            precisions.append(result.get('precision_macro_test', 0))
            recalls.append(result.get('recall_macro_test', 0))
    
    if not models:
        print(f"   ⚠️ No valid results for plotting")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison - Filtered Features', fontsize=16, fontweight='bold')
    
    metrics = [
        (accuracies, 'Accuracy', axes[0, 0]),
        (f1_macros, 'F1-Score (Macro)', axes[0, 1]),
        (f1_weighteds, 'F1-Score (Weighted)', axes[0, 2]),
        (precisions, 'Precision (Macro)', axes[1, 0]),
        (recalls, 'Recall (Macro)', axes[1, 1])
    ]
    
    for values, metric_name, ax in metrics:
        bars = ax.bar(models, values, color='lightcoral', alpha=0.8)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Remove extra subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    
    # Save plot
    filename = 'model_performance_comparison.png'
    filepath = os.path.join(plots_folder, filename)
    
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    Model comparison plot saved: {filename}")
    except Exception as e:
        print(f"    Error saving comparison plot: {e}")
    
    plt.close()

# === SAVE RESULTS WITH ENHANCED PRECISION ===
def save_hyperparameters(results):
    """
    Save hyperparameters to JSON and CSV files with enhanced metrics
    """
    print(f"\n Saving hyperparameters with enhanced metrics...")
    if OPTIMIZE_HYPERPARAMETERS == True:
        print(f"⚙️ Mode: Optimized Parameters")
    elif OPTIMIZE_HYPERPARAMETERS == False:
        print(f"⚙️ Mode: Fixed Parameters")
    elif OPTIMIZE_HYPERPARAMETERS == "default":
        print(f"⚙️ Mode: Default Parameters")
    
    # Prepare data for JSON
    hyperparams_json = {}
    hyperparams_csv = []
    
    for model_name, result in results.items():
        if not result or 'best_params' not in result:
            continue
        
        best_params = result['best_params']
        
        if (OPTIMIZE_HYPERPARAMETERS != True or 
            best_params in ["N/A (Base Model)", "N/A", "Fixed Parameters (No Optimization)", "Scikit-Learn Default Parameters"]):
            
            param_mode = ""
            if OPTIMIZE_HYPERPARAMETERS == False:
                param_mode = "Fixed parameters - no optimization"
            elif OPTIMIZE_HYPERPARAMETERS == "default":
                param_mode = "Scikit-learn default parameters"
            else:
                param_mode = "Base model - no optimization"
            
            hyperparams_json[model_name] = {
                'hyperparameters': param_mode,
                'performance_metrics': {
                    'accuracy': result.get('accuracy_test', 'N/A'),
                    'f1_macro': result.get('f1_macro_test', 'N/A'),
                    'f1_weighted': result.get('f1_weighted_test', 'N/A'),
                    'precision_macro': result.get('precision_macro_test', 'N/A'),
                    'recall_macro': result.get('recall_macro_test', 'N/A')
                }
            }
            
            # Prepare CSV data
            opt_mode = ""
            if OPTIMIZE_HYPERPARAMETERS == False:
                opt_mode = "Fixed Parameters"
            elif OPTIMIZE_HYPERPARAMETERS == "default":
                opt_mode = "Default Parameters"
            else:
                opt_mode = "No Hyperparameter Space"
            
            base_row = {
                'Model': model_name,
                'CV_Score': result.get('best_cv_search_score', 'N/A'),
                'Test_Accuracy': result.get('accuracy_test', 'N/A'),
                'Test_F1_Macro': result.get('f1_macro_test', 'N/A'),
                'Test_F1_Weighted': result.get('f1_weighted_test', 'N/A'),
                'Test_Precision_Weighted': result.get('precision_weighted_test', 'N/A'),
                'Test_Recall_Weighted': result.get('recall_weighted_test', 'N/A'),
                'Test_Precision_Macro': result.get('precision_macro_test', 'N/A'),
                'Test_Recall_Macro': result.get('recall_macro_test', 'N/A'),
                'Optimization_Mode': opt_mode
            }
            hyperparams_csv.append(base_row)
            continue
        
        # Process hyperparameters for optimized models
        processed_params = {}
        if isinstance(best_params, dict):
            for param_name, param_value in best_params.items():
                if callable(param_value):
                    if param_name == 'weights':
                        processed_params[param_name] = f"custom_function_{id(param_value)}"
                    else:
                        processed_params[param_name] = str(param_value)
                elif isinstance(param_value, np.ndarray):
                    processed_params[param_name] = param_value.tolist()
                elif isinstance(param_value, (np.integer, np.floating)):
                    processed_params[param_name] = param_value.item()
                else:
                    processed_params[param_name] = param_value
        
        hyperparams_json[model_name] = {
            'hyperparameters': processed_params,
            'performance_metrics': {
                'cv_score': result.get('best_cv_search_score', 'N/A'),
                'accuracy': result.get('accuracy_test', 'N/A'),
                'f1_macro': result.get('f1_macro_test', 'N/A'),
                'f1_weighted': result.get('f1_weighted_test', 'N/A'),
                'precision_weighted': result.get('precision_weighted_test', 'N/A'),
                'recall_weighted': result.get('recall_weighted_test', 'N/A'),
                'precision_macro': result.get('precision_macro_test', 'N/A'),
                'recall_macro': result.get('recall_macro_test', 'N/A')
            }
        }
        
        # Prepare CSV data with enhanced metrics
        base_row = {
            'Model': model_name,
            'CV_Score': result.get('best_cv_search_score', 'N/A'),
            'Test_Accuracy': result.get('accuracy_test', 'N/A'),
            'Test_F1_Macro': result.get('f1_macro_test', 'N/A'),
            'Test_F1_Weighted': result.get('f1_weighted_test', 'N/A'),
            'Test_Precision_Weighted': result.get('precision_weighted_test', 'N/A'),
            'Test_Recall_Weighted': result.get('recall_weighted_test', 'N/A'),
            'Test_Precision_Macro': result.get('precision_macro_test', 'N/A'),
            'Test_Recall_Macro': result.get('recall_macro_test', 'N/A'),
            'Optimization_Mode': 'Hyperparameter Optimization'
        }
        
        if isinstance(best_params, dict):
            for param_name, param_value in best_params.items():
                if callable(param_value):
                    if param_name == 'weights':
                        param_value = f"custom_function_{id(param_value)}"
                    else:
                        param_value = str(param_value)
                elif isinstance(param_value, (list, tuple)):
                    param_value = str(param_value)
                elif isinstance(param_value, np.ndarray):
                    param_value = str(param_value.tolist())
                
                base_row[f'param_{param_name}'] = param_value
        
        hyperparams_csv.append(base_row)
    
    # Save JSON
    json_filename = 'hyperparameters_filtered_features.json'
    json_filepath = os.path.join(output_folder, json_filename)
    
    try:
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(hyperparams_json, f, indent=2, ensure_ascii=False)
        print(f"    Hyperparameters JSON saved: {json_filename}")
    except Exception as e:
        print(f"    Error saving JSON: {e}")
    
    # Save CSV with high precision
    if hyperparams_csv:
        csv_filename = 'hyperparameters_filtered_features.csv'
        csv_filepath = os.path.join(output_folder, csv_filename)
        
        try:
            df = pd.DataFrame(hyperparams_csv)
            # Organize columns
            info_cols = ['Model', 'CV_Score', 'Test_Accuracy', 'Test_F1_Macro', 'Test_F1_Weighted', 
                        'Test_Precision_Weighted', 'Test_Recall_Weighted', 'Test_Precision_Macro', 'Test_Recall_Macro', 'Optimization_Mode']
            param_cols = [col for col in df.columns if col.startswith('param_')]
            ordered_cols = info_cols + sorted(param_cols)
            df = df[ordered_cols]
            
            # Save with high precision for all decimal values
            df.to_csv(csv_filepath, index=False, float_format='%.12f')
            print(f"    Hyperparameters CSV saved: {csv_filename}")
            print(f"    Enhanced metrics saved with 12 decimal places precision")
        except Exception as e:
            print(f"    Error saving CSV: {e}")

def save_shap_rankings(shap_rankings, class_names):
    """
    Save detailed SHAP rankings to CSV with high precision
    """
    print(f"\n Saving SHAP rankings with high precision...")
    
    if not shap_rankings:
        print(f"   ⚠️ No SHAP rankings to save")
        return
    
    # Detailed SHAP rankings with per-class scores
    detailed_rankings = []
    
    for model_name, ranking_data in shap_rankings.items():
        if 'feature_ranking' not in ranking_data:
            continue
        
        for rank, (feature_name, overall_importance) in enumerate(ranking_data['feature_ranking'], 1):
            base_row = {
                'Model': model_name,
                'Rank': rank,
                'Feature': feature_name,
                'Overall_SHAP_Importance': overall_importance
            }
            
            # Add per-class SHAP scores
            shap_per_class = ranking_data.get('shap_values_per_class', {})
            for class_name in class_names:
                if class_name in shap_per_class:
                    # Find the feature index
                    if 'overall_importance' in ranking_data:
                        feature_names_list = list(ranking_data['overall_importance'].keys())
                        if feature_name in feature_names_list:
                            feature_idx = feature_names_list.index(feature_name)
                            if feature_idx < len(shap_per_class[class_name]):
                                base_row[f'SHAP_Score_Class_{class_name}'] = shap_per_class[class_name][feature_idx]
                            else:
                                base_row[f'SHAP_Score_Class_{class_name}'] = 0.0
                        else:
                            base_row[f'SHAP_Score_Class_{class_name}'] = 0.0
                    else:
                        base_row[f'SHAP_Score_Class_{class_name}'] = 0.0
                else:
                    base_row[f'SHAP_Score_Class_{class_name}'] = 0.0
            
            detailed_rankings.append(base_row)
    
    if not detailed_rankings:
        print(f"   ⚠️ No ranking data to save")
        return
    
    # Create DataFrame and save
    df_rankings = pd.DataFrame(detailed_rankings)
    
    # Organize columns
    base_cols = ['Model', 'Rank', 'Feature', 'Overall_SHAP_Importance']
    class_cols = [col for col in df_rankings.columns if col.startswith('SHAP_Score_Class_')]
    ordered_cols = base_cols + sorted(class_cols)
    df_rankings = df_rankings[ordered_cols]
    
    filename = 'shap_complete_feature_rankings.csv'
    filepath = os.path.join(output_folder, filename)
    
    try:
        # Save with high precision (12 decimal places)
        df_rankings.to_csv(filepath, index=False, float_format='%.12f')
        print(f"    Detailed SHAP rankings saved: {filename}")
        print(f"       {len(df_rankings)} feature rankings saved with 12 decimal places precision")
        print(f"       Includes per-class SHAP scores for all {len(class_names)} classes")
        
        # Show top features per model
        print(f"\n    Top 5 features per model:")
        for model in df_rankings['Model'].unique():
            model_data = df_rankings[df_rankings['Model'] == model].head(5)
            print(f"      {model}:")
            for _, row in model_data.iterrows():
                print(f"         {row['Rank']:2d}. {row['Feature']:<25} (Score: {row['Overall_SHAP_Importance']:.4f})")
        
    except Exception as e:
        print(f"   Error saving SHAP rankings: {e}")

def generate_summary_report(results, X_train, X_test, class_names):
    print(f"\n Generating summary report...")
    
    # Find best model
    best_model = None
    best_accuracy = 0
    
    for model_name, result in results.items():
        if result and result.get('accuracy_test', 0) > best_accuracy:
            best_accuracy = result.get('accuracy_test', 0)
            best_model = model_name
    
    summary = {
        "pipeline_info": {
            "script_version": script_version,
            "execution_timestamp": pd.Timestamp.now().isoformat(),
            "scenario": "Filtered Features with SHAP at End + Focused Hyperparameter Space",
            "dataset_path": DATASET_PATH,
            "hyperparameter_optimization": OPTIMIZE_HYPERPARAMETERS
        },
        "dataset_info": {
            "num_features": X_train.shape[1],
            "num_classes": len(class_names),
            "class_names": class_names,
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "feature_selection_method": "ReliefF + ANOVA + Correlation"
        },
        "model_performance": {
            "models_trained": len(results),
            "best_model": {
                "name": best_model,
                "accuracy": best_accuracy
            } if best_model else None,
            "all_results": {
                model_name: {
                    "accuracy": result.get('accuracy_test', 0),
                    "f1_macro": result.get('f1_macro_test', 0),
                    "f1_weighted": result.get('f1_weighted_test', 0),
                    "precision_macro": result.get('precision_macro_test', 0),
                    "recall_macro": result.get('recall_macro_test', 0)
                }
                for model_name, result in results.items()
                if result
            }
        },
        "configuration": {
            "skip_lightgbm": SKIP_LIGHTGBM,
            "intensive_search": INTENSIVE_SEARCH,
            "optimize_hyperparameters": OPTIMIZE_HYPERPARAMETERS,
            "hyperparameter_strategy": (
                "Focused search with 60-75% probability of optimal values" if OPTIMIZE_HYPERPARAMETERS == True 
                else "Fixed pre-optimized parameters" if OPTIMIZE_HYPERPARAMETERS == False
                else "Scikit-learn default parameters"
            )
        }
    }
    
    # Save summary
    filename = 'pipeline_summary_report.json'
    filepath = os.path.join(output_folder, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"    Summary report saved: {filename}")
    except Exception as e:
        print(f"    Error saving summary: {e}")
    
    return summary

# === MAIN EXECUTION ===
def main():
    """
    Main execution function - SHAP analysis moved to the END
    """
    print(f"\n{'='*60}")
    print(f" SCENARIO 3: FILTERED FEATURES PIPELINE (FOCUSED HYPERSPACE)")
    if OPTIMIZE_HYPERPARAMETERS == True:
        print(f" HYPERPARAMETER MODE: OPTIMIZATION")
    elif OPTIMIZE_HYPERPARAMETERS == False:
        print(f" HYPERPARAMETER MODE: FIXED PARAMETERS")
    elif OPTIMIZE_HYPERPARAMETERS == "default":
        print(f" HYPERPARAMETER MODE: DEFAULT PARAMETERS")
    print(f"{'='*60}")
    
    # Load dataset
    X_train, X_test, y_train, y_test, le = load_dataset()
    
    if X_train is None:
        print(f" Failed to load dataset. Please check the data files.")
        print(f"   Make sure ReliefF_ANOVA feature selection has been run first.")
        return
    
    class_names = list(le.classes_)
    print(f"\n Target classes ({len(class_names)}): {class_names}")
    
    # ===== PHASE 1: TRAINING AND BASIC EVALUATION =====
    print(f"\n{'='*60}")
    print(f" PHASE 1: MODEL TRAINING AND EVALUATION")
    if OPTIMIZE_HYPERPARAMETERS == True:
        print(f" Mode: Hyperparameter Optimization")
    elif OPTIMIZE_HYPERPARAMETERS == False:
        print(f" Mode: Fixed Parameters")
    elif OPTIMIZE_HYPERPARAMETERS == "default":
        print(f" Mode: Default Parameters")
    print(f"{'='*60}")
    
    results, predictions, trained_models = train_evaluate_models(
        X_train, y_train, X_test, y_test, class_names
    )
    
    if not results:
        print(f" No models were successfully trained.")
        return
    
    # ===== PHASE 2: VISUALIZATIONS AND REPORTING =====
    print(f"\n{'='*60}")
    print(f" PHASE 2: VISUALIZATIONS AND REPORTING")
    print(f"{'='*60}")
    
    # Generate visualizations
    plot_confusion_matrices(predictions, class_names)
    plot_model_comparison(results)
    
    # Save results
    save_hyperparameters(results)
    
    # Generate summary
    summary = generate_summary_report(results, X_train, X_test, class_names)
    
    # Print intermediate summary
    print(f"\n{'='*60}")
    print(f" PHASE 2 COMPLETED - BASIC ANALYSIS DONE")
    print(f"{'='*60}")
    print(f" Results folder: {output_folder}")
    print(f" Plots folder: {plots_folder}")
    
    if summary and summary['model_performance']['best_model']:
        best = summary['model_performance']['best_model']
        print(f"\n Best Model: {best['name']} (Accuracy: {best['accuracy']:.4f})")
    
    print(f"\n Dataset Summary:")
    print(f"    Filtered features: {summary['dataset_info']['num_features']}")
    print(f"    Feature selection: ReliefF + ANOVA + Correlation")
    print(f"    Final dataset: {summary['dataset_info']['train_samples']} train + {summary['dataset_info']['test_samples']} test")
    if OPTIMIZE_HYPERPARAMETERS == True:
        print(f"    Hyperparameter mode: Optimization")
    elif OPTIMIZE_HYPERPARAMETERS == False:
        print(f"    Hyperparameter mode: Fixed Parameters")
    elif OPTIMIZE_HYPERPARAMETERS == "default":
        print(f"    Hyperparameter mode: Default Parameters")
    
    print(f"\n Files generated so far:")
    print(f"    Hyperparameters (JSON + CSV) with precision/recall macro")
    print(f"    Confusion matrices")
    print(f"    Model comparison plots")
    print(f"    Summary report")
    if OPTIMIZE_HYPERPARAMETERS == True:
        print(f"    Used focused hyperparameter search with bias towards optimal values")
    elif OPTIMIZE_HYPERPARAMETERS == False:
        print(f"    Used fixed pre-optimized parameters")
    elif OPTIMIZE_HYPERPARAMETERS == "default":
        print(f"    Used scikit-learn default parameters")
    
    # ===== PHASE 3: COMPREHENSIVE SHAP ANALYSIS =====
    print(f"\n{'='*60}")
    print(f" PHASE 3: COMPREHENSIVE SHAP ANALYSIS")
    print(f"{'='*60}")
    print(f" Performing detailed SHAP analysis on trained models...")
    print(f" This is the most computationally intensive part")
    
    if trained_models:
        print(f"\n Models available for SHAP analysis:")
        for i, model_name in enumerate(trained_models.keys(), 1):
            print(f"   {i}. {model_name}")
        
        # Generate SHAP analysis (FINAL STEP)
        shap_rankings = generate_shap_analysis(trained_models, X_test, X_train, class_names)
        
        # Save SHAP results (FINAL OUTPUTS)
        save_shap_rankings(shap_rankings, class_names)
        
        print(f"\n{'='*60}")
        print(f" SHAP ANALYSIS")
        print(f"{'='*60}")
        print(f" SHAP analysis folder: {shap_folder}")
        
        if shap_rankings:
            print(f"\n SHAP analysis completed for {len(shap_rankings)} models:")
            for model_name, ranking_data in shap_rankings.items():
                if 'feature_ranking' in ranking_data:
                    top_3_features = [f[0] for f in ranking_data['feature_ranking'][:3]]
                    print(f"   - {model_name}: Top 3 features: {top_3_features}")
        else:
            print(f"    No SHAP rankings were generated")
    else:
        print(f"    No trained models available for SHAP analysis")
    
    # ===== FINAL SUMMARY =====
    print(f"\n{'='*60}")
    print(f" COMPLETE PIPELINE")
    print(f"{'='*60}")
    print(f" All results saved in: {output_folder}")
    print(f" Plots: {plots_folder}")
    print(f" SHAP analysis: {shap_folder}")
    
    print(f"\n ALL generated files:")
    print(f"    Model Performance:")
    print(f"      - Hyperparameters (JSON + CSV) with precision/recall macro")
    print(f"      - Confusion matrices")
    print(f"      - Model comparison plots")
    print(f"      - Summary report")
    print(f"      SHAP Analysis (Generated Last):")
    print(f"      - Complete SHAP feature rankings with 12 decimal precision")
    print(f"      - SHAP summary plots per class")
    print(f"      - SHAP importance plots per model")
    
    print(f"\n Pipeline execution order:")
    print(f"   1. Model training and evaluation")
    print(f"   2. Basic visualizations and reporting")
    print(f"   3. Comprehensive SHAP analysis (FINAL)")
    
    print(f"\n Enhanced metrics saved with 12 decimal places:")
    print(f"    Precision Macro and Recall Macro included in all CSVs")
    print(f"    All SHAP values saved with high precision")
    
    print(f"\n Feature selection method: ReliefF + ANOVA + Correlation")
    if OPTIMIZE_HYPERPARAMETERS == True:
        print(f" Hyperparameter strategy: Focused optimization search")
    elif OPTIMIZE_HYPERPARAMETERS == False:
        print(f" Hyperparameter strategy: Fixed pre-optimized parameters")
    elif OPTIMIZE_HYPERPARAMETERS == "default":
        print(f" Hyperparameter strategy: Scikit-learn default parameters")
    print(f" Compare with other scenarios to evaluate feature selection impact!")

if __name__ == "__main__":
    main()