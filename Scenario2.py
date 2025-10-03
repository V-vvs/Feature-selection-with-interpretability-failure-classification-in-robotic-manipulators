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
from scipy.stats import randint, uniform, loguniform, spearmanr

import lightgbm as lgb
import shap
import warnings

warnings.filterwarnings('ignore')

plt.ioff()
try:
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['PYTHONUNBUFFERED'] = '1'
except Exception:
    pass

import plotly.io as pio
pio.renderers.default = "png"

script_version = "ML_Pipeline_Scenario2_SHAP_Only"
print(f" Starting ML Pipeline {script_version}")
print(" SCENARIO 2: Full Feature Dataset from Preprocessed_Data/")
print(" Step 1: Train all models with full features")
print(" Step 2: SHAP analysis on final models")

# === CONFIGURATION ===
SKIP_LIGHTGBM = False
INTENSIVE_SEARCH = True
DATASET_PATH = r'./Preprocessed_Data'

print(f" Configuration: SKIP_LIGHTGBM={SKIP_LIGHTGBM}, INTENSIVE_SEARCH={INTENSIVE_SEARCH}")
print(f" Dataset path: {DATASET_PATH}")

# === OUTPUT CONFIGURATION ===
output_folder = f'ML_Results_Scenario2_{script_version}'
os.makedirs(output_folder, exist_ok=True)
plots_folder = os.path.join(output_folder, 'Plots')
os.makedirs(plots_folder, exist_ok=True)
shap_folder = os.path.join(output_folder, 'SHAP_Analysis')
os.makedirs(shap_folder, exist_ok=True)
lightgbm_folder = os.path.join(output_folder, 'LightGBM_Analysis')
os.makedirs(lightgbm_folder, exist_ok=True)

# === HYPERPARAMETER SPACE ===
def get_hyperparameter_space():
    return {
        'KNN': {
            'n_neighbors': [1, 2, 3, 5, 7, 9, 15, 21],
            'weights': ['uniform', 'distance'] + [
                lambda d: 1 / (d + 1e-6),
                lambda d: np.exp(-d),
            ],
            'metric': ['euclidean', 'manhattan', 'cosine'],
            'p': [1, 1.2, 1.5, 2, 2.5, 3],
            'algorithm': ['kd_tree']
        },
        'SVM': {
            'C': uniform(0.1, 1000),
            'gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, 10)),
            'kernel': ['rbf', 'linear'],
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
            'n_estimators': randint(50, 1000),
            'learning_rate': loguniform(0.005, 0.1),
            'num_leaves': randint(5, 50),
            'max_depth': randint(3, 7),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'reg_alpha': loguniform(0.01, 1000.0),
            'reg_lambda': loguniform(0.01, 1000.0),
            'min_child_samples': randint(10, 50),
            'min_data_in_leaf': randint(15, 50),
            'class_weight': ['balanced'],
            'verbosity': [-1],
            'feature_fraction': uniform(0.5, 0.5),
            'bagging_fraction': uniform(0.5, 0.5),
            'bagging_freq': randint(1, 10)
        }
    }

# === MODEL CONFIGURATION ===
def get_search_iterations(n_features, n_samples):
    feature_ratio = n_features / n_samples
    print(f" Dataset info: {n_samples} samples, {n_features} features")
    print(f" Feature/Sample ratio: {feature_ratio:.3f}")
    
    iterations = {
        'KNN': 60,
        'SVM': 150,
        'MLP': 300,
        'LightGBM': 100
    }
    
    if feature_ratio > 0.4:
        print(" High dimensionality detected - using conservative search")
    
    return iterations

def get_optimized_models(X_train, skip_lightgbm=False):
    print(f"\n Setting up models for full features scenario")
    
    base_models = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True, random_state=42),
        'MLP': MLPClassifier(random_state=42, early_stopping=True, n_iter_no_change=15, max_iter=1000),
    }
    
    if not skip_lightgbm:
        base_models['LightGBM'] = lgb.LGBMClassifier(
            random_state=42, verbosity=-1, n_jobs=1,
            force_row_wise=True, class_weight='balanced'
        )
    
    param_distributions = get_hyperparameter_space()
    
    search_iterations = get_search_iterations(X_train.shape[1], X_train.shape[0])
    cv_folds = 10
    
    optimized_models = {}
    for name, base_model in base_models.items():
        if name not in param_distributions:
            optimized_models[name] = clone(base_model)
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
        
        print(f"  {name}: {search_iterations[name]} iterations, {cv_folds}-fold CV")
    
    return optimized_models

# === DATASET LOADING ===
def load_dataset():
    print(f"\n Loading full features dataset...")
    
    train_file = os.path.join(DATASET_PATH, 'dataset_train_transformed.csv')
    test_file = os.path.join(DATASET_PATH, 'dataset_test_transformed.csv')
    
    if not os.path.exists(train_file):
        print(f" Train file not found: {train_file}")
        return None, None, None, None, None
    
    if not os.path.exists(test_file):
        print(f" Test file not found: {test_file}")
        return None, None, None, None, None
    
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        print(f" Train dataset: {train_df.shape}")
        print(f" Test dataset: {test_df.shape}")
        
        if 'label' not in train_df.columns or 'label' not in test_df.columns:
            print(f" 'label' column not found in datasets")
            return None, None, None, None, None
        
        X_train = train_df.drop('label', axis=1)
        y_train_labels = train_df['label']
        X_test = test_df.drop('label', axis=1)
        y_test_labels = test_df['label']
        
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_labels)
        y_test = le.transform(y_test_labels)
        
        class_names = list(le.classes_)
        
        print(f" Features: {X_train.shape[1]}")
        print(f" Classes: {class_names}")
        print(f" Train samples: {X_train.shape[0]}")
        print(f" Test samples: {X_test.shape[0]}")
        
        unique_labels, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(le.inverse_transform(unique_labels), counts))
        print(f" Class distribution: {class_dist}")
        
        return X_train, X_test, y_train, y_test, le
        
    except Exception as e:
        print(f" Error loading datasets: {e}")
        return None, None, None, None, None

# === TRAINING AND EVALUATION ===
def train_evaluate_models(X_train, y_train, X_test, y_test, class_names):
    print(f"\n STEP 1: Training and evaluating models with ALL features...")
    
    feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
    
    models = get_optimized_models(X_train, skip_lightgbm=SKIP_LIGHTGBM)
    
    results = {}
    predictions = {}
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\n     Training {model_name}...")
        sys.stdout.flush()
        
        if model_name == 'LightGBM':
            print(f"   LightGBM optimization started (this may take a few minutes)...")
            sys.stdout.flush()
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index) if feature_names else X_train_scaled
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index) if feature_names else X_test_scaled
        
        try:
            if isinstance(model, RandomizedSearchCV):
                model.fit(X_train_scaled_df, y_train)
                best_model = model.best_estimator_
                best_params = model.best_params_
                best_cv_score = model.best_score_
            else:
                model.fit(X_train_scaled_df, y_train)
                best_model = model
                best_params = "N/A (Base Model)"
                y_train_pred = best_model.predict(X_train_scaled_df)
                best_cv_score = f1_score(y_train, y_train_pred, average='macro')
            
            y_pred = best_model.predict(X_test_scaled_df)
            
            trained_models[model_name] = {
                'model': best_model,
                'scaler': scaler,
                'best_params': best_params,
                'best_cv_score': best_cv_score,
                'feature_names': feature_names,
                'training_data': {
                    'X_train_scaled': X_train_scaled_df,
                    'X_test_scaled': X_test_scaled_df
                }
            }
            
            predictions[model_name] = {
                'y_true': y_test,
                'y_pred': y_pred
            }
            
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            
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
                'classification_report_test': classification_report(y_test, y_pred, target_names=class_names, zero_division=0, output_dict=False),
                'features_used': len(feature_names) if feature_names else X_train.shape[1],
                'model_version': 'full_features'
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

# === LIGHTGBM FEATURE IMPORTANCE EXTRACTION ===
def extract_lightgbm_feature_importance(trained_models, feature_names):
    print(f"\n STEP 2: Extracting Native LightGBM Feature Importance...")
    
    lightgbm_rankings = {}
    
    for model_name, model_assets in trained_models.items():
        if model_name != 'LightGBM' or not model_assets or 'model' not in model_assets:
            continue
            
        model = model_assets['model']
        model_feature_names = model_assets.get('feature_names', feature_names)
        
        if not isinstance(model, lgb.LGBMClassifier):
            print(f"     {model_name} is not LightGBM, skipping...")
            continue
            
        try:
            print(f" Calculating importance for {model_name}...")
            
            importance_methods = {
                'gain': 'gain',
                'split': 'split'
            }
            
            method_results = {}
            
            for method_name, method_type in importance_methods.items():
                try:
                    if hasattr(model, 'feature_importances_'):
                        sklearn_importance = model.feature_importances_
                        
                        feature_importance_pairs = [
                            (model_feature_names[i] if model_feature_names else f"Feature_{i}", sklearn_importance[i])
                            for i in range(len(sklearn_importance))
                        ]
                        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                        
                        method_results[f'sklearn_{method_name}'] = {
                            'ranking': feature_importance_pairs,
                            'method': 'sklearn_feature_importances_',
                            'description': 'LightGBM native importance via sklearn interface'
                        }
                        
                        print(f"      Sklearn importance: {len(feature_importance_pairs)} features")
                    
                    if hasattr(model, 'booster_'):
                        native_importance = model.booster_.feature_importance(
                            importance_type=method_type
                        )
                        
                        native_pairs = [
                            (model_feature_names[i] if model_feature_names else f"Feature_{i}", native_importance[i])
                            for i in range(len(native_importance))
                        ]
                        native_pairs.sort(key=lambda x: x[1], reverse=True)
                        
                        method_results[f'native_{method_name}'] = {
                            'ranking': native_pairs,
                            'method': f'booster_.feature_importance(importance_type="{method_type}")',
                            'description': f'LightGBM native {method_type} importance'
                        }
                        
                        print(f"      Native {method_type} importance: {len(native_pairs)} features")
                        
                except Exception as e:
                    print(f"      Error in {method_name}: {e}")
                    continue
            
            if method_results:
                lightgbm_rankings[model_name] = method_results
                
                for method_name, result in method_results.items():
                    top_5 = [f[0] for f in result['ranking'][:5]]
                    print(f"       Top 5 ({method_name}): {top_5}")
            
        except Exception as e:
            print(f"     Error extracting importance from {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return lightgbm_rankings

# === SHAP ANALYSIS ===
def generate_shap_analysis(all_trained_models, X_test, X_train, class_names):
    print(f"\n STEP 3: Starting SHAP Analysis on ALL models...")
    print(f"     Analyzing ALL {X_test.shape[0]} test samples")
    print(f"     Using ALL {X_train.shape[0]} training samples as background")
    
    if X_test.empty:
        print(f"    Test data is empty. Skipping SHAP analysis.")
        return {}
    
    feature_names = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else None
    shap_rankings = {}
    
    for model_name, model_assets in all_trained_models.items():
        if not model_assets or 'model' not in model_assets:
            continue
            
        print(f"\n     Calculating SHAP for {model_name}...")
        
        model = model_assets['model']
        scaler = model_assets['scaler']
        model_feature_names = model_assets.get('feature_names', feature_names)
        
        try:
            X_test_current = X_test[model_feature_names]
            X_train_current = X_train[model_feature_names]
                
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_current), columns=model_feature_names)
            X_train_scaled = pd.DataFrame(scaler.transform(X_train_current), columns=model_feature_names)
            
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
                        print(f"      Unexpected SHAP array shape: {shap_values.shape}")
                        continue
                else:
                    print(f"      Unexpected SHAP format: {shap_values.shape}")
                    continue
            else:
                print(f"      Unrecognized SHAP format: {type(shap_values)}")
                continue
            
            mean_abs_shaps_per_class = []
            for i, class_name in enumerate(class_names):
                if i < len(shap_values_processed):
                    class_shap_values = shap_values_processed[i]
                    mean_abs_shap = np.abs(class_shap_values).mean(axis=0)
                    mean_abs_shaps_per_class.append(mean_abs_shap)
                    print(f"        Class '{class_name}': calculated from {class_shap_values.shape[0]} samples")
            
            if not mean_abs_shaps_per_class:
                print(f"       No valid SHAP values for {model_name}")
                continue
            
            overall_importance = np.mean(mean_abs_shaps_per_class, axis=0)
            feature_importance = {
                (model_feature_names[j] if model_feature_names else f"Feature_{j}"): overall_importance[j] 
                for j in range(len(overall_importance))
            }
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            shap_rankings[model_name] = {
                'feature_ranking': sorted_features,
                'shap_values_per_class': {class_names[i]: mean_abs_shaps_per_class[i] for i in range(len(mean_abs_shaps_per_class))},
                'overall_importance': feature_importance,
                'model_version': model_assets.get('selection_info', {}).get('model_version', 'unknown')
            }
            
            print(f"       SHAP analysis completed for {model_name}")
            print(f"        Features analyzed: {len(model_feature_names)}")
            print(f"        Top 5 features: {[f[0] for f in sorted_features[:5]]}")
            
            generate_shap_plots(shap_values_processed, X_test_current, model_feature_names, class_names, model_name, sorted_features)
            
        except Exception as e:
            print(f"      Error in SHAP calculation for {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return shap_rankings

def generate_shap_plots(shap_values_processed, X_test, feature_names, class_names, model_name, sorted_features):
    try:
        print(f"           Generating SHAP plots for {model_name}...")
        
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
                    print(f"             Error creating SHAP summary for class {class_name}: {e}")
                
                plt.close()
        
        plt.figure(figsize=(12, max(8, len(sorted_features[:20]) * 0.4)))
        top_features = sorted_features[:20]
        feature_names_plot = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]
        
        y_pos = np.arange(len(feature_names_plot))
        plt.barh(y_pos, importance_values, color='skyblue', alpha=0.8)
        plt.yticks(y_pos, feature_names_plot)
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Feature Importance (SHAP) - {model_name}')
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()
        
        for i, v in enumerate(importance_values):
            plt.text(v + max(importance_values) * 0.01, i, f'{v:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        filename = f'SHAP_Importance_{model_name}.png'
        filepath = os.path.join(shap_folder, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"             SHAP importance plot saved: {filename}")
        plt.close()
        
    except Exception as e:
        print(f"           Error generating SHAP plots for {model_name}: {e}")

# === SAVE LIGHTGBM RANKINGS ===
def save_lightgbm_rankings(lightgbm_rankings, output_folder):
    print(f"\n Saving LightGBM Rankings...")
    
    if not lightgbm_rankings:
        print(f"     No LightGBM rankings to save")
        return
    
    all_rankings_csv = []
    
    for model_name, methods in lightgbm_rankings.items():
        for method_name, method_data in methods.items():
            ranking = method_data['ranking']
            method_description = method_data['description']
            
            for rank, (feature_name, importance_score) in enumerate(ranking, 1):
                all_rankings_csv.append({
                    'Model': model_name,
                    'Method': method_name,
                    'Method_Description': method_description,
                    'Rank': rank,
                    'Feature': feature_name,
                    'Importance_Score': importance_score,
                    'Normalized_Score': importance_score / ranking[0][1] if ranking[0][1] != 0 else 0
                })
    
    if all_rankings_csv:
        df_rankings = pd.DataFrame(all_rankings_csv)
        
        csv_filename = 'lightgbm_feature_rankings.csv'
        csv_filepath = os.path.join(output_folder, csv_filename)
        
        try:
            df_rankings.to_csv(csv_filepath, index=False, float_format='%.12f')
            print(f"     CSV saved: {csv_filename}")
            print(f"       {len(df_rankings)} rankings saved")
        except Exception as e:
            print(f"     Error saving CSV: {e}")
    
    json_filename = 'lightgbm_feature_rankings.json'
    json_filepath = os.path.join(output_folder, json_filename)
    
    try:
        json_data = {}
        for model_name, methods in lightgbm_rankings.items():
            json_data[model_name] = {}
            for method_name, method_data in methods.items():
                json_data[model_name][method_name] = {
                    'method': method_data['method'],
                    'description': method_data['description'],
                    'feature_ranking': [
                        {'feature': feature, 'importance': float(importance)}
                        for feature, importance in method_data['ranking']
                    ]
                }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"     JSON saved: {json_filename}")
    except Exception as e:
        print(f"     Error saving JSON: {e}")

# === COMPARE RANKINGS ===
def compare_feature_rankings(lightgbm_rankings, shap_rankings, output_folder):
    print(f"\n Comparing Rankings: LightGBM vs SHAP...")
    
    if not lightgbm_rankings or not shap_rankings:
        print(f"     Insufficient data for comparison")
        return
    
    comparisons = []
    
    if 'LightGBM' in lightgbm_rankings and 'LightGBM' in shap_rankings:
        lgb_methods = lightgbm_rankings['LightGBM']
        shap_ranking = shap_rankings['LightGBM']['feature_ranking']
        
        for method_name, method_data in lgb_methods.items():
            lgb_ranking = method_data['ranking']
            
            lgb_top20 = [f[0] for f in lgb_ranking[:20]]
            shap_top20 = [f[0] for f in shap_ranking[:20]]
            
            overlap_features = set(lgb_top20) & set(shap_top20)
            overlap_percentage = len(overlap_features) / min(20, len(shap_top20)) * 100
            
            common_features = set(f[0] for f in lgb_ranking) & set(f[0] for f in shap_ranking)
            
            if len(common_features) >= 10:
                lgb_ranks = {f[0]: i+1 for i, f in enumerate(lgb_ranking)}
                shap_ranks = {f[0]: i+1 for i, f in enumerate(shap_ranking)}
                
                common_lgb_ranks = [lgb_ranks[f] for f in common_features]
                common_shap_ranks = [shap_ranks[f] for f in common_features]
                
                correlation, p_value = spearmanr(common_lgb_ranks, common_shap_ranks)
            else:
                correlation, p_value = np.nan, np.nan
            
            comparisons.append({
                'LightGBM_Model': 'LightGBM',
                'SHAP_Model': 'LightGBM',
                'LightGBM_Method': method_name,
                'Top20_Overlap_Count': len(overlap_features),
                'Top20_Overlap_Percentage': overlap_percentage,
                'Rank_Correlation': correlation,
                'Correlation_P_Value': p_value,
                'Common_Features_Count': len(common_features),
                'LightGBM_Top5': lgb_top20[:5],
                'SHAP_Top5': shap_top20[:5],
                'Overlapping_Top5': list(set(lgb_top20[:5]) & set(shap_top20[:5]))
            })
            
            print(f"     LightGBM ({method_name}) vs SHAP (LightGBM):")
            print(f"      Overlap Top-20: {overlap_percentage:.1f}%")
            print(f"      Rank correlation: {correlation:.3f}" if not np.isnan(correlation) else "      Rank correlation: N/A")
            
    if comparisons:
        df_comparisons = pd.DataFrame(comparisons)
        
        filename = 'lightgbm_vs_shap_comparison.csv'
        filepath = os.path.join(output_folder, filename)
        
        try:
            df_comparisons.to_csv(filepath, index=False, float_format='%.12f')
            print(f"   Comparison saved: {filename}")
        except Exception as e:
            print(f"   Error saving comparison: {e}")

# === PLOT LIGHTGBM IMPORTANCE ===
def plot_lightgbm_importance(lightgbm_rankings, plots_folder):
    print(f"\n Generating LightGBM Feature Importance plots...")
    
    if not lightgbm_rankings:
        print(f"  No data to plot")
        return
    
    for model_name, methods in lightgbm_rankings.items():
        for method_name, method_data in methods.items():
            try:
                ranking = method_data['ranking']
                method_description = method_data['description']
                
                top_20 = ranking[:20]
                features = [f[0] for f in top_20]
                importances = [f[1] for f in top_20]
                
                plt.figure(figsize=(12, max(8, len(features) * 0.4)))
                
                y_pos = np.arange(len(features))
                bars = plt.barh(y_pos, importances, color='lightcoral', alpha=0.8)
                
                plt.yticks(y_pos, features)
                plt.xlabel('Feature Importance Score')
                plt.title(f'LightGBM Feature Importance - {method_description}\nModel: {model_name}')
                plt.grid(axis='x', alpha=0.3)
                plt.gca().invert_yaxis()
                
                for i, (bar, value) in enumerate(zip(bars, importances)):
                    plt.text(value + max(importances) * 0.01, i, f'{value:.1f}', 
                                va='center', fontsize=9)
                
                plt.tight_layout()
                
                filename = f'LightGBM_Importance_{model_name}_{method_name}.png'
                filepath = os.path.join(plots_folder, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"     Plot saved: {filename}")
                
            except Exception as e:
                print(f"     Error creating plot for {model_name} {method_name}: {e}")

# === CONFUSION MATRIX PLOTS ===
def plot_confusion_matrices(all_predictions, class_names):
    print(f"\n Generating confusion matrices...")
    
    n_models = len([name for name, pred_data in all_predictions.items() 
                    if pred_data and 'y_true' in pred_data and 'y_pred' in pred_data])
    
    if n_models == 0:
        print(f"    No valid predictions available")
        return
    
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
    
    fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
    
    plot_idx = 0
    for model_name, pred_data in all_predictions.items():
        if not pred_data or 'y_true' not in pred_data or 'y_pred' not in pred_data:
            continue
        
        y_true = pred_data['y_true']
        y_pred = pred_data['y_pred']
        
        if len(y_true) == 0 or len(y_pred) == 0:
            continue
        
        ax = axes[plot_idx]
        
        cm = confusion_matrix(y_true, y_pred)
        
        cm_sum_axis1 = cm.sum(axis=1)[:, np.newaxis]
        cm_percent = np.zeros_like(cm, dtype=float)
        np.divide(cm.astype('float'), cm_sum_axis1, out=cm_percent, where=cm_sum_axis1!=0)
        cm_percent *= 100
        
        annot_data = [[f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)"
                        for j in range(cm.shape[1])]
                        for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=annot_data, fmt='s', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar_kws={'label': 'No. of Samples'})
        
        acc_val = accuracy_score(y_true, y_pred)
        f1_macro_val = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted_val = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        ax.set_title(f'{model_name}\nAcc: {acc_val:.3f} | F1-M: {f1_macro_val:.3f} | F1-W: {f1_weighted_val:.3f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        
        plot_idx += 1
    
    for i in range(plot_idx, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    
    filename = 'confusion_matrices_all_models.png'
    filepath = os.path.join(plots_folder, filename)
    
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"     Confusion matrices saved: {filename}")
    except Exception as e:
        print(f"     Error saving confusion matrices: {e}")
    
    plt.close()

# === RESULTS VISUALIZATION ===
def plot_model_comparison(all_results):
    print(f"\n Generating model performance plots...")
    
    if not all_results:
        print(f"    No results available for plotting")
        return
    
    models = []
    accuracies = []
    f1_macros = []
    f1_weighteds = []
    precisions = []
    recalls = []
    feature_counts = []
    model_types = []
    
    for model_name, result in all_results.items():
        if result:
            models.append(model_name)
            accuracies.append(result.get('accuracy_test', 0))
            f1_macros.append(result.get('f1_macro_test', 0))
            f1_weighteds.append(result.get('f1_weighted_test', 0))
            precisions.append(result.get('precision_macro_test', 0))
            recalls.append(result.get('recall_macro_test', 0))
            feature_counts.append(result.get('features_used', 0))
            model_types.append(result.get('model_version', 'unknown'))
    
    if not models:
        print(f"     No valid results for plotting")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Model Performance - All Models', fontsize=16, fontweight='bold')
    
    colors = ['skyblue'] * len(models)
    
    metrics = [
        (accuracies, 'Accuracy', axes[0, 0]),
        (f1_macros, 'F1-Score (Macro)', axes[0, 1]),
        (f1_weighteds, 'F1-Score (Weighted)', axes[0, 2]),
        (precisions, 'Precision (Macro)', axes[1, 0]),
        (recalls, 'Recall (Macro)', axes[1, 1])
    ]
    
    for values, metric_name, ax in metrics:
        bars = ax.bar(models, values, color=colors, alpha=0.8)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax = axes[1, 2]
    bars = ax.bar(models, feature_counts, color=colors, alpha=0.8)
    ax.set_title('Number of Features Used', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Count', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, feature_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(feature_counts)*0.01,
                f'{value}', ha='center', va='bottom', fontsize=9)
    
    # Remove any existing legends
    try:
        if hasattr(fig, 'legends') and fig.legends:
            for legend in fig.legends:
                legend.remove()
        
        for ax in axes.flat:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    except Exception as e:
        print(f"     Warning: Could not remove legends: {e}")
    
    plt.tight_layout()
    
    filename = 'model_performance_all_models.png'
    filepath = os.path.join(plots_folder, filename)
    
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"     Model performance plot saved: {filename}")
    except Exception as e:
        print(f"     Error saving comparison plot: {e}")
    
    plt.close()

# === SAVE RESULTS WITH ENHANCED PRECISION ===
def save_hyperparameters(all_results):
    print(f"\n Saving hyperparameters for all models...")
    
    hyperparams_json = {}
    hyperparams_csv = []
    
    for model_name, result in all_results.items():
        if not result or 'best_params' not in result:
            continue
        
        best_params = result['best_params']
        
        if best_params == "N/A (Base Model)" or best_params == "N/A":
            hyperparams_json[model_name] = {
                'hyperparameters': "Base model - no optimization",
                'performance_metrics': {
                    'accuracy': result.get('accuracy_test', 'N/A'),
                    'f1_macro': result.get('f1_macro_test', 'N/A'),
                    'f1_weighted': result.get('f1_weighted_test', 'N/A'),
                    'precision_macro': result.get('precision_macro_test', 'N/A'),
                    'recall_macro': result.get('recall_macro_test', 'N/A')
                },
                'model_info': {
                    'features_used': result.get('features_used', 'N/A'),
                    'model_version': result.get('model_version', 'unknown')
                }
            }
            continue
        
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
            },
            'model_info': {
                'features_used': result.get('features_used', 'N/A'),
                'model_version': result.get('model_version', 'unknown')
            }
        }
        
        base_row = {
            'Model': model_name,
            'Model_Version': result.get('model_version', 'unknown'),
            'Features_Used': result.get('features_used', 'N/A'),
            'CV_Score': result.get('best_cv_search_score', 'N/A'),
            'Test_Accuracy': result.get('accuracy_test', 'N/A'),
            'Test_F1_Macro': result.get('f1_macro_test', 'N/A'),
            'Test_F1_Weighted': result.get('f1_weighted_test', 'N/A'),
            'Test_Precision_Weighted': result.get('precision_weighted_test', 'N/A'),
            'Test_Recall_Weighted': result.get('recall_weighted_test', 'N/A'),
            'Test_Precision_Macro': result.get('precision_macro_test', 'N/A'),
            'Test_Recall_Macro': result.get('recall_macro_test', 'N/A')
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
    
    json_filename = 'hyperparameters_all_models.json'
    json_filepath = os.path.join(output_folder, json_filename)
    
    try:
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(hyperparams_json, f, indent=2, ensure_ascii=False)
        print(f"     Hyperparameters JSON saved: {json_filename}")
    except Exception as e:
        print(f"     Error saving JSON: {e}")
    
    if hyperparams_csv:
        csv_filename = 'hyperparameters_all_models.csv'
        csv_filepath = os.path.join(output_folder, csv_filename)
        
        try:
            df = pd.DataFrame(hyperparams_csv)
            info_cols = ['Model', 'Model_Version', 'Features_Used', 'CV_Score', 'Test_Accuracy', 
                        'Test_F1_Macro', 'Test_F1_Weighted', 'Test_Precision_Weighted', 
                        'Test_Recall_Weighted', 'Test_Precision_Macro', 'Test_Recall_Macro']
            param_cols = [col for col in df.columns if col.startswith('param_')]
            ordered_cols = info_cols + sorted(param_cols)
            df = df[ordered_cols]
            
            # Save with high precision for all decimal values
            df.to_csv(csv_filepath, index=False, float_format='%.12f')
            print(f"     Hyperparameters CSV saved: {csv_filename}")
            print(f"     Metrics saved with 12 decimal places precision")
        except Exception as e:
            print(f"     Error saving CSV: {e}")

def save_shap_rankings(shap_rankings):
    print(f"\n Saving SHAP rankings for all models...")
    
    if not shap_rankings:
        print(f"    No SHAP rankings to save")
        return
    
    all_rankings = []
    
    for model_name, ranking_data in shap_rankings.items():
        if 'feature_ranking' not in ranking_data:
            continue
        
        model_version = ranking_data.get('model_version', 'unknown')
        
        for rank, (feature_name, importance_score) in enumerate(ranking_data['feature_ranking'], 1):
            all_rankings.append({
                'Model': model_name,
                'Model_Version': model_version,
                'Rank': rank,
                'Feature': feature_name,
                'SHAP_Importance': importance_score
            })
    
    if not all_rankings:
        print(f"    No ranking data to save")
        return
    
    df_rankings = pd.DataFrame(all_rankings)
    
    filename = 'shap_feature_rankings_all_models.csv'
    filepath = os.path.join(output_folder, filename)
    
    try:
        df_rankings.to_csv(filepath, index=False, float_format='%.12f')
        print(f"    SHAP rankings saved: {filename}")
        print(f"     {len(df_rankings)} feature rankings saved with 12 decimal places precision")
    except Exception as e:
        print(f"     Error saving SHAP rankings: {e}")

def generate_summary_report(all_results, X_train, X_test, class_names):
    print(f"\n Generating comprehensive summary report...")
    
    best_model = None
    best_acc = 0
    
    for model_name, result in all_results.items():
        if result:
            acc = result.get('accuracy_test', 0)
            if acc > best_acc:
                best_acc = acc
                best_model = model_name
    
    summary = {
        "pipeline_info": {
            "script_version": script_version,
            "execution_timestamp": pd.Timestamp.now().isoformat(),
            "scenario": "Full Features + SHAP Analysis Only",
            "dataset_path": DATASET_PATH
        },
        "dataset_info": {
            "num_features": X_train.shape[1],
            "num_classes": len(class_names),
            "class_names": class_names,
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0]
        },
        "model_performance": {
            "models_trained": len(all_results),
            "best_model": {
                "name": best_model,
                "accuracy": best_acc
            } if best_model else None,
            "all_results": {
                model_name: {
                    "accuracy": result.get('accuracy_test', 0),
                    "f1_macro": result.get('f1_macro_test', 0),
                    "f1_weighted": result.get('f1_weighted_test', 0),
                    "precision_macro": result.get('precision_macro_test', 0),
                    "recall_macro": result.get('recall_macro_test', 0),
                    "features_used": result.get('features_used', 0),
                    "model_version": result.get('model_version', 'unknown')
                }
                for model_name, result in all_results.items()
                if result
            }
        },
        "configuration": {
            "skip_lightgbm": SKIP_LIGHTGBM,
            "intensive_search": INTENSIVE_SEARCH
        }
    }
    
    filename = 'pipeline_summary_report_complete.json'
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
    print(f"\n{'='*80}")
    print(f" SIMPLIFIED 2-STEP PIPELINE: FULL TRAINING + SHAP ANALYSIS")
    print(f"{'='*80}")
    
    X_train, X_test, y_train, y_test, le = load_dataset()
    
    if X_train is None:
        print(f" Failed to load dataset. Please check the data files.")
        return
    
    class_names = list(le.classes_)
    print(f"\n Target classes ({len(class_names)}): {class_names}")
    
    # === STEP 1: Train all models with FULL features ===
    print(f"\n{'='*60}")
    print(f" STEP 1: TRAINING ALL MODELS WITH FULL FEATURES")
    print(f"{'='*60}")
    
    results, predictions, trained_models = train_evaluate_models(
        X_train, y_train, X_test, y_test, class_names
    )
    
    if not results:
        print(f" No models were successfully trained.")
        return
    
    # === STEP 2: LightGBM Feature Importance (for analysis only) ===
    print(f"\n{'='*60}")
    print(f" STEP 2: LIGHTGBM FEATURE IMPORTANCE (ANALYSIS ONLY)")
    print(f"{'='*60}")
    
    feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
    lightgbm_rankings = extract_lightgbm_feature_importance(trained_models, feature_names)
    
    # === STEP 3: SHAP Analysis on ALL models ===
    print(f"\n{'='*60}")
    print(f" STEP 3: SHAP ANALYSIS ON ALL MODELS")
    print(f"{'='*60}")
    
    shap_rankings = generate_shap_analysis(trained_models, X_test, X_train, class_names)
    
    # === STEP 4: Analysis and Visualization ===
    print(f"\n{'='*60}")
    print(f" STEP 4: GENERATING ANALYSIS AND VISUALIZATIONS")
    print(f"{'='*60}")
    
    if lightgbm_rankings and shap_rankings:
        compare_feature_rankings(lightgbm_rankings, shap_rankings, output_folder)
    
    plot_confusion_matrices(predictions, class_names)
    plot_model_comparison(results)
    
    if lightgbm_rankings:
        plot_lightgbm_importance(lightgbm_rankings, plots_folder)
    
    save_hyperparameters(results)
    save_shap_rankings(shap_rankings)
    
    if lightgbm_rankings:
        save_lightgbm_rankings(lightgbm_rankings, output_folder)
    
    summary = generate_summary_report(results, X_train, X_test, class_names)
    
    # === FINAL SUMMARY ===
    print(f"\n{'='*80}")
    print(f" SIMPLIFIED PIPELINE FINISHED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f" Results folder: {output_folder}")
    print(f" Plots folder: {plots_folder}")
    print(f" SHAP analysis: {shap_folder}")
    print(f" LightGBM analysis: {lightgbm_folder}")
    
    if summary:
        best_model_info = summary['model_performance'].get('best_model')
        if best_model_info:
            print(f"\n Best Model: {best_model_info['name']} (Accuracy: {best_model_info['accuracy']:.4f})")
    
    print(f"\n Generated files:")
    print(f"    - Hyperparameters for all models (JSON + CSV) with precision/recall macro")
    print(f"    - SHAP feature rankings for all models")
    print(f"    - LightGBM feature rankings (analysis only)")
    print(f"    - LightGBM vs SHAP comparison")
    print(f"    - Confusion matrices")
    print(f"    - Model performance plots")
    print(f"    - SHAP analysis plots")
    print(f"    - LightGBM importance plots")
    print(f"    - Comprehensive summary report")
    
    print(f"\n 2-step simplified pipeline executed successfully!")
    print(f" Step 1: Complete training with full features ✓")
    print(f" Step 2: SHAP analysis on all models ✓")
    print(f" All metrics saved with 12 decimal places precision ✓")

if __name__ == "__main__":
    main()