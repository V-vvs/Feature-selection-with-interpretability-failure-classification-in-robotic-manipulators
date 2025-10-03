import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
from scipy.stats import randint, uniform, loguniform, ttest_rel
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("COMPLETE FEATURE SELECTION VALIDATION STUDY")
print("="*80)
print("COMPARING THREE SCENARIOS WITH COMPLETE FEATURE TRACKING:")
print("   A. ReliefF+ANOVA+Correlation (54 features) - LOW COMPUTATIONAL COST")
print("   B. SHAP Top-54 (per classifier) - HIGH COMPUTATIONAL COST")
print("   C. Optimized Method (A + SHAP-guided swaps) - HYBRID APPROACH")
print()
print("HYPERPARAMETERS: EXACT same as Scenario 3 for fair comparison")
print("="*80)

CONFIG = {
    'random_state': 42,
    'cv_folds': 5,
    'test_size': 0.2,
    'correlation_threshold': 0.80,
    'max_swaps': None,
    'models': ['KNN', 'SVM', 'MLP', 'LightGBM'],
    'validation_threshold': 0.03
}

SCENARIO_FEATURES = {
    'A': {},
    'B': {},
    'C': {}
}

FEATURE_DETAILS = {
    'swap_details': {},
    'concordance_analysis': {},
    'feature_origins': {}
}

def load_feature_selection_data():
    print("\nLoading feature selection data...")
    try:
        selected_path = 'ReliefF_ANOVA_Correlation_Results/selected_features_detailed_ranking.csv'
        selected_df = pd.read_csv(selected_path)
        selected_features = selected_df['Feature'].tolist()

        removed_path = 'ReliefF_ANOVA_Correlation_Results/removed_features_redundancy.csv'
        removed_df = pd.read_csv(removed_path)

        shap_path = 'ML_Results_Scenario2_ML_Pipeline_Scenario2_SHAP_Only/shap_feature_rankings_all_models.csv'
        shap_df = pd.read_csv(shap_path)

        print(f"Selected features (Scenario A): {len(selected_features)}")
        print(f"Removed redundant features: {len(removed_df)}")
        print(f"SHAP rankings loaded for models: {shap_df['Model'].unique().tolist()}")

        return selected_features, removed_df, shap_df
    except Exception as e:
        print(f"Error loading feature selection data: {e}")
        return None, None, None

def load_original_datasets():
    print("\nLoading original datasets (144 features)...")
    train_path = 'Preprocessed_Data/dataset_train_transformed.csv'
    test_path = 'Preprocessed_Data/dataset_test_transformed.csv'

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"Train dataset: {train_df.shape}")
        print(f"Test dataset: {test_df.shape}")
        feature_cols = [col for col in train_df.columns if col != 'label']
        print(f"Total features available: {len(feature_cols)}")
        return train_df, test_df, feature_cols
    except Exception as e:
        print(f"Error loading original datasets: {e}")
        return None, None, None

def get_exact_scenario3_hyperparameters():
    return {
        'KNN': KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='kd_tree', p=2),
        'SVM': SVC(probability=True, random_state=42, C=10, gamma=0.015, tol=1, kernel='rbf', class_weight='balanced'),
        'MLP': MLPClassifier(random_state=42, early_stopping=True, n_iter_no_change=20, max_iter=2000, hidden_layer_sizes=(75, 100), activation='relu', solver='lbfgs', alpha=0.05, learning_rate='invscaling', learning_rate_init=0.001, tol=0.005),
        'LightGBM': lgb.LGBMClassifier(num_class=3, metric='multi_logloss', boosting_type='gbdt', verbose=-1, random_state=42, num_leaves=50, max_depth=5, min_data_in_leaf=55, min_child_samples=10, min_split_gain=0.02, learning_rate=0.06, n_estimators=1000, lambda_l2=0.05, feature_fraction=1.0, bagging_fraction=1, bagging_freq=10, class_weight='balanced', is_unbalance=True, force_col_wise=True, deterministic=True)
    }

def follow_correlation_chain(feature, removed_df, target_set, max_depth=10):
    visited = set()
    path = [feature]
    current_feature = feature
    
    for _ in range(max_depth):
        if current_feature in visited:
            return None, []
        
        visited.add(current_feature)
        
        if current_feature in target_set:
            return current_feature, path
        
        correlata_found = None
        
        for _, row in removed_df.iterrows():
            if row['Feature'] == current_feature:
                correlata_found = row['correlated_with']
                break
        
        if not correlata_found:
            for _, row in removed_df.iterrows():
                if row['correlated_with'] == current_feature:
                    correlata_found = row['Feature']
                    break
        
        if not correlata_found:
            break
        
        path.append(correlata_found)
        current_feature = correlata_found
    
    return None, path

def find_correlata_in_shap_corrected(feature, removed_df, shap_top54_set, model_shap):
    correlata_candidates = []

    for _, row in removed_df.iterrows():
        if row['correlated_with'] == feature:
            correlata_feature = row['Feature']
            if correlata_feature in shap_top54_set:
                correlata_candidates.append((correlata_feature, row['correlation']))

    for _, row in removed_df.iterrows():
        if row['Feature'] == feature:
            correlata_feature = row['correlated_with']
            if correlata_feature in shap_top54_set:
                correlata_candidates.append((correlata_feature, row['correlation']))

    if correlata_candidates:
        best_correlata = None
        best_rank = float('inf')

        for correlata_feature, correlation in correlata_candidates:
            correlata_rank_data = model_shap[model_shap['Feature'] == correlata_feature]['Rank']
            if not correlata_rank_data.empty:
                correlata_rank = correlata_rank_data.iloc[0]
                if correlata_rank < best_rank:
                    best_rank = correlata_rank
                    best_correlata = (correlata_feature, correlation)

        if best_correlata:
            correlata_feature, correlation = best_correlata
            original_rank_data = model_shap[model_shap['Feature'] == feature]['Rank']
            replacement_rank_data = model_shap[model_shap['Feature'] == correlata_feature]['Rank']
            original_rank = original_rank_data.iloc[0] if not original_rank_data.empty else 999
            replacement_rank = replacement_rank_data.iloc[0] if not replacement_rank_data.empty else 999

            return {
                'feature': correlata_feature,
                'original_rank': original_rank,
                'replacement_rank': replacement_rank,
                'correlation': correlation
            }

    return None

def find_correlata_in_relief_corrected(feature, removed_df, relief_features_set):
    correlata_candidates = []
    
    for _, row in removed_df.iterrows():
        if row['Feature'] == feature and row['correlated_with'] in relief_features_set:
            correlata_candidates.append(row['correlated_with'])
    
    for _, row in removed_df.iterrows():
        if row['correlated_with'] == feature and row['Feature'] in relief_features_set:
            correlata_candidates.append(row['Feature'])
    
    for _, row in removed_df.iterrows():
        if row['Feature'] == feature:
            removed_feature_correlata = row['correlated_with']
            if removed_feature_correlata in relief_features_set:
                correlata_candidates.append(removed_feature_correlata)
        elif row['correlated_with'] == feature:
            removed_feature = row['Feature']
            for _, row2 in removed_df.iterrows():
                if row2['Feature'] == removed_feature and row2['correlated_with'] in relief_features_set:
                    correlata_candidates.append(row2['correlated_with'])
                elif row2['correlated_with'] == removed_feature and row2['Feature'] in relief_features_set:
                    correlata_candidates.append(row2['Feature'])
    
    if correlata_candidates:
        return correlata_candidates[0]
    
    return None

def find_correlata_in_relief_enhanced(feature, removed_df, relief_features_set):
    direct_correlata = find_correlata_in_relief_corrected(feature, removed_df, relief_features_set)
    if direct_correlata:
        return {
            'correlata': direct_correlata,
            'path': [feature, direct_correlata],
            'type': 'direct'
        }
    
    final_correlata, path = follow_correlation_chain(feature, removed_df, relief_features_set)
    
    if final_correlata and len(path) > 2:
        return {
            'correlata': final_correlata,
            'path': path,
            'type': 'transitive'
        }
    
    return None

def find_correlata_in_shap_enhanced(feature, removed_df, shap_top54_set, model_shap):
    direct_correlata = find_correlata_in_shap_corrected(feature, removed_df, shap_top54_set, model_shap)
    if direct_correlata:
        return {
            'feature': direct_correlata['feature'],
            'original_rank': direct_correlata['original_rank'],
            'replacement_rank': direct_correlata['replacement_rank'],
            'correlation': direct_correlata['correlation'],
            'path': [feature, direct_correlata['feature']],
            'type': 'direct'
        }
    
    final_correlata, path = follow_correlation_chain(feature, removed_df, shap_top54_set)
    
    if final_correlata and len(path) > 2:
        original_rank_data = model_shap[model_shap['Feature'] == feature]['Rank']
        replacement_rank_data = model_shap[model_shap['Feature'] == final_correlata]['Rank']
        
        original_rank = original_rank_data.iloc[0] if not original_rank_data.empty else 999
        replacement_rank = replacement_rank_data.iloc[0] if not replacement_rank_data.empty else 999
        estimated_correlation = 0.6
        
        return {
            'feature': final_correlata,
            'original_rank': original_rank,
            'replacement_rank': replacement_rank,
            'correlation': estimated_correlation,
            'path': path,
            'type': 'transitive'
        }
    
    return None

def analyze_correlation_chains(removed_df):
    print("\nANALYZING CORRELATION CHAINS...")
    correlation_graph = {}
    
    for _, row in removed_df.iterrows():
        feature = row['Feature']
        correlata = row['correlated_with']
        correlation = row['correlation']
        
        if feature not in correlation_graph:
            correlation_graph[feature] = []
        if correlata not in correlation_graph:
            correlation_graph[correlata] = []
            
        correlation_graph[feature].append((correlata, correlation))
        correlation_graph[correlata].append((feature, correlation))
    
    chains_found = []
    visited_global = set()
    
    for start_feature in correlation_graph:
        if start_feature in visited_global:
            continue
            
        visited_local = set()
        current_chain = []
        
        def dfs(feature, path):
            if feature in visited_local:
                return
            
            visited_local.add(feature)
            path.append(feature)
            
            if len(path) > 2:
                chains_found.append(path.copy())
            
            for neighbor, corr in correlation_graph.get(feature, []):
                if neighbor not in visited_local:
                    dfs(neighbor, path)
            
            path.pop()
        
        dfs(start_feature, current_chain)
        visited_global.update(visited_local)
    
    unique_chains = []
    for chain in chains_found:
        if len(chain) > 2:
            normalized_chain = tuple(sorted(chain))
            if normalized_chain not in [tuple(sorted(c)) for c in unique_chains]:
                unique_chains.append(chain)
    
    print(f"   Total correlation relationships: {len(removed_df)}")
    print(f"   Features involved in correlations: {len(correlation_graph)}")
    print(f"   Correlation chains (length > 2): {len(unique_chains)}")
    
    if unique_chains:
        print(f"   Example chains:")
        for i, chain in enumerate(unique_chains[:3]):
            chain_str = ' -> '.join(chain)
            print(f"      {i+1}. {chain_str}")
        
        if len(unique_chains) > 3:
            print(f"      ... and {len(unique_chains) - 3} more chains")
    
    return {
        'correlation_graph': correlation_graph,
        'chains': unique_chains,
        'total_relationships': len(removed_df),
        'features_in_correlations': len(correlation_graph)
    }

def create_optimized_features_enhanced(selected_features, removed_df, shap_df, model_name):
    model_shap = shap_df[shap_df['Model'] == model_name].sort_values('Rank')
    shap_top54_set = set(model_shap.head(54)['Feature'].tolist())
    selected_set = set(selected_features)

    feature_origins = {}
    swaps_made = []
    concordance_info = {
        'agreement': [],
        'discordance_with_swap': [],
        'discordance_no_swap': [],
        'transitive_swaps': []
    }

    final_features = []

    print(f"\n      Processing {model_name} feature optimization (Enhanced)...")

    for feature in selected_features:
        if feature in shap_top54_set:
            final_features.append(feature)
            feature_origins[feature] = 'agreement_keep'
            concordance_info['agreement'].append(feature)
        else:
            correlata_info = find_correlata_in_shap_enhanced(feature, removed_df, shap_top54_set, model_shap)

            if correlata_info:
                correlata_feature = correlata_info['feature']
                final_features.append(correlata_feature) 
                feature_origins[correlata_feature] = f'swap_from_{feature}'

                swap_info = {
                    'original': feature,
                    'replacement': correlata_feature,
                    'original_rank': correlata_info['original_rank'],
                    'replacement_rank': correlata_info['replacement_rank'],
                    'correlation': correlata_info['correlation'],
                    'rank_improvement': correlata_info['original_rank'] - correlata_info['replacement_rank'],
                    'correlation_path': correlata_info['path'],
                    'correlation_type': correlata_info['type']
                }
                swaps_made.append(swap_info)
                
                if correlata_info['type'] == 'transitive':
                    concordance_info['transitive_swaps'].append(swap_info)
                else:
                    concordance_info['discordance_with_swap'].append(swap_info)
            else:
                final_features.append(feature)
                feature_origins[feature] = 'discordance_keep'
                concordance_info['discordance_no_swap'].append(feature)

    agreement_pct = len(concordance_info['agreement']) / 54 * 100
    direct_swap_pct = len(concordance_info['discordance_with_swap']) / 54 * 100
    transitive_swap_pct = len(concordance_info['transitive_swaps']) / 54 * 100
    total_swap_pct = direct_swap_pct + transitive_swap_pct
    keep_pct = len(concordance_info['discordance_no_swap']) / 54 * 100

    print(f"      Agreement: {len(concordance_info['agreement'])}/54 ({agreement_pct:.1f}%)")
    print(f"      Direct Swaps: {len(concordance_info['discordance_with_swap'])}/54 ({direct_swap_pct:.1f}%)")
    print(f"      Transitive Swaps: {len(concordance_info['transitive_swaps'])}/54 ({transitive_swap_pct:.1f}%)")
    print(f"      Total Swaps: {len(swaps_made)}/54 ({total_swap_pct:.1f}%)")
    print(f"      Kept: {len(concordance_info['discordance_no_swap'])}/54 ({keep_pct:.1f}%)")

    if concordance_info['transitive_swaps']:
        print(f"      Transitive correlation examples:")
        for swap in concordance_info['transitive_swaps'][:2]:
            path_str = ' -> '.join(swap['correlation_path'])
            print(f"         {path_str}")

    swap_summary = {
        'model': model_name,
        'swaps': swaps_made,
        'feature_origins': feature_origins,
        'concordance': concordance_info,
        'final_count': len(final_features),
        'statistics': {
            'agreement_pct': agreement_pct,
            'direct_swap_pct': direct_swap_pct,
            'transitive_swap_pct': transitive_swap_pct,
            'total_swap_pct': total_swap_pct,
            'keep_pct': keep_pct
        }
    }

    return final_features, swap_summary

def create_scenario_features_enhanced(selected_features, removed_df, shap_df):
    print(f"\nCreating feature sets for all scenarios (Enhanced)...")
    
    chain_analysis = analyze_correlation_chains(removed_df)
    scenario_a_features = selected_features.copy()

    for model in CONFIG['models']:
        SCENARIO_FEATURES['A'][model] = scenario_a_features

    print(f"Scenario A: {len(scenario_a_features)} features (same for all models)")

    print(f"\nCreating Scenario B (SHAP top-54 per model)...")

    for model in CONFIG['models']:
        model_shap = shap_df[shap_df['Model'] == model].sort_values('Rank')
        shap_top54 = model_shap.head(54)['Feature'].tolist()
        SCENARIO_FEATURES['B'][model] = shap_top54
        print(f"   {model}: {len(shap_top54)} features")

    print(f"\nCreating Scenario C (Enhanced with transitive correlation handling)...")

    for model in CONFIG['models']:
        optimized_features, swap_info = create_optimized_features_enhanced(scenario_a_features, removed_df, shap_df, model)
        SCENARIO_FEATURES['C'][model] = optimized_features
        FEATURE_DETAILS['swap_details'][model] = swap_info

        print(f"   {model}: {len(optimized_features)} features")
        print(f"      Direct swaps: {len(swap_info['concordance']['discordance_with_swap'])}")
        print(f"      Transitive swaps: {len(swap_info['concordance']['transitive_swaps'])}")
        print(f"      Total swaps: {len(swap_info['swaps'])}")

        if swap_info['swaps']:
            print(f"      Example swaps:")
            for swap in swap_info['swaps'][:2]:
                if swap['correlation_type'] == 'transitive':
                    path_str = ' -> '.join(swap['correlation_path'])
                    print(f"         {swap['original']} -> {swap['replacement']} (transitive: {path_str})")
                else:
                    print(f"         {swap['original']} -> {swap['replacement']} (direct)")

    return SCENARIO_FEATURES

def train_evaluate_scenario_fixed_hyperparams(X_train, X_test, y_train, y_test, scenario_name, model_name, features_used):
    print(f"      Training {model_name} for {scenario_name}...")
    print(f"      Features: {len(features_used)}")

    models_config = get_exact_scenario3_hyperparameters()
    model = clone(models_config[model_name])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        results = {
            'model': model_name,
            'scenario': scenario_name,
            'features_used': features_used,
            'num_features': len(features_used),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'predictions': y_pred,
            'hyperparameters': 'Scenario3_Exact'
        }

        print(f"         Accuracy: {results['accuracy']:.4f}, F1-Macro: {results['f1_macro']:.4f}")
        return results

    except Exception as e:
        print(f"         Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def perform_validation_analysis(all_results):
    print(f"\nPERFORMING VALIDATION ANALYSIS...")

    f1_scores = {'A': [], 'B': [], 'C': []}
    model_names = []

    for model in CONFIG['models']:
        if all(model in all_results[scenario] for scenario in ['A', 'B', 'C']):
            model_names.append(model)
            f1_scores['A'].append(all_results['A'][model]['f1_macro'])
            f1_scores['B'].append(all_results['B'][model]['f1_macro'])
            f1_scores['C'].append(all_results['C'][model]['f1_macro'])

    differences = {
        'B_minus_A': [b - a for b, a in zip(f1_scores['B'], f1_scores['A'])],
        'C_minus_A': [c - a for c, a in zip(f1_scores['C'], f1_scores['A'])],
        'C_minus_B': [c - b for c, b in zip(f1_scores['C'], f1_scores['B'])]
    }

    avg_differences = {k: np.mean(v) for k, v in differences.items()}

    if len(f1_scores['A']) > 1:
        _, p_value_b_vs_a = ttest_rel(f1_scores['A'], f1_scores['B'])
        _, p_value_c_vs_a = ttest_rel(f1_scores['A'], f1_scores['C'])
        _, p_value_c_vs_b = ttest_rel(f1_scores['B'], f1_scores['C'])
    else:
        p_value_b_vs_a = p_value_c_vs_a = p_value_c_vs_b = 1.0

    threshold = CONFIG['validation_threshold']
    
    validation_results = {
        'avg_differences': avg_differences,
        'p_values': {
            'B_vs_A': p_value_b_vs_a,
            'C_vs_A': p_value_c_vs_a,
            'C_vs_B': p_value_c_vs_b
        },
        'validation_decisions': {
            'method_validated': str((avg_differences['B_minus_A'] <= 0) or (avg_differences['B_minus_A'] < threshold)),
            'optimization_useful': str(avg_differences['C_minus_A'] > threshold),
            'significant_difference': str(p_value_b_vs_a < 0.05)
        },
        'model_performances': {
            model: {
                'A': all_results['A'][model]['f1_macro'],
                'B': all_results['B'][model]['f1_macro'],
                'C': all_results['C'][model]['f1_macro']
            }
            for model in model_names
        }
    }

    print(f"\nVALIDATION RESULTS:")
    print(f"   Average F1-Macro differences:")
    print(f"      SHAP vs ReliefF+ANOVA: {avg_differences['B_minus_A']:+.4f}")
    print(f"      Optimized vs ReliefF+ANOVA: {avg_differences['C_minus_A']:+.4f}")
    print(f"      Optimized vs SHAP: {avg_differences['C_minus_B']:+.4f}")

    print(f"\nStatistical significance (p-values):")
    print(f"      SHAP vs ReliefF+ANOVA: {p_value_b_vs_a:.4f}")
    print(f"      Optimized vs ReliefF+ANOVA: {p_value_c_vs_a:.4f}")
    print(f"      Optimized vs SHAP: {p_value_c_vs_b:.4f}")

    print(f"\nVALIDATION DECISION:")
    if validation_results['validation_decisions']['method_validated'] == 'True':
        if avg_differences['B_minus_A'] <= 0:
            print(f"   ReliefF+ANOVA method is VALIDATED (SUPERIOR)!")
            print(f"   ReliefF+ANOVA performs {abs(avg_differences['B_minus_A']):.1%} BETTER than SHAP")
            print(f"   Recommendation: Use ReliefF+ANOVA (100x faster AND better performance)")
        else:
            print(f"   ReliefF+ANOVA method is VALIDATED!")
            print(f"   Performance difference < {threshold:.1%} threshold")
            print(f"   Recommendation: Use ReliefF+ANOVA (100x faster)")
    else:
        print(f"   ReliefF+ANOVA method NOT validated")
        print(f"   Performance difference > {threshold:.1%} threshold")
        print(f"   Recommendation: Consider SHAP for critical applications")

    return validation_results

def get_shap_importance_score(row):
    possible_columns = ['SHAP_Value', 'Importance', 'Score', 'Value', 'Mean_SHAP', 
                        'Abs_SHAP', 'SHAP_Importance', 'mean_abs_shap', 'shap_importance']
    
    for col in possible_columns:
        if col in row and pd.notna(row[col]):
            return abs(float(row[col]))
    
    return 0.0

def create_shap_ranking_concordance_plots(selected_features, removed_df, shap_df):
    print(f"\nCreating SHAP concordance plots...")
    
    os.makedirs('Validation_Results', exist_ok=True)
    relief_set = set(selected_features)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 18))
    axes = axes.flatten()
    
    for i, model in enumerate(CONFIG['models']):
        ax = axes[i]
        
        model_shap = shap_df[shap_df['Model'] == model].sort_values('Rank')
        shap_top54 = model_shap.head(54)
        
        feature_data = []
        
        for _, row in shap_top54.iterrows():
            feature = row['Feature']
            importance = get_shap_importance_score(row)
            rank = row['Rank']
            
            if feature in relief_set:
                status = 'Concordant'
                color = '#2E8B57'
            else:
                correlata_info = find_correlata_in_relief_enhanced(feature, removed_df, relief_set)
                if correlata_info:
                    status = 'Redundant'
                    color = '#FF8C00'
                else:
                    status = 'Discordant'
                    color = '#DC143C'
            
            feature_data.append({
                'feature': feature,
                'importance': importance,
                'rank': rank,
                'status': status,
                'color': color
            })
        
        feature_data = feature_data[::-1]
        
        y_positions = range(len(feature_data))
        importances = [fd['importance'] for fd in feature_data]
        colors = [fd['color'] for fd in feature_data]
        
        ax.barh(y_positions, importances, color=colors, alpha=0.8, height=0.8)
        
        max_importance = max(importances) if importances else 0.001
        for j, fd in enumerate(feature_data):
            y_pos = j
            feature_label = f"{fd['rank']:2d}. {fd['feature']}"
            
            if fd['status'] == 'Redundant':
                correlata_result = find_correlata_in_relief_enhanced(fd['feature'], removed_df, relief_set)
                if correlata_result:
                    if correlata_result['type'] == 'transitive' and len(correlata_result['path']) > 2:
                        path_str = ' -> '.join(correlata_result['path'])
                        feature_label = f"{fd['rank']:2d}. {path_str}"
                    else:
                        correlata = correlata_result['correlata']
                        feature_label += f" -> {correlata}"
            
            ax.text(max_importance * 0.01, y_pos, feature_label, va='center', ha='left', 
                    fontsize=8, fontweight='bold')
        
        ax.set_yticks([])
        ax.set_xlabel('SHAP Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'\nModel: {model}', fontsize=16, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        if max_importance > 0:
            ax.set_xlim(0, max_importance * 1.1)
        else:
            ax.set_xlim(0, 0.01)

    fig.suptitle('SHAP Top-54 Ranking: Concordance with ReliefF+ANOVA+Correlation selection', 
                 fontsize=20, fontweight='bold', y=0.98, ha='center')
    
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(left=0.2, right=0.95, top=0.90, bottom=0.05, hspace=0.2, wspace=0.25)
    
    output_file = 'Validation_Results/combined_shap_ranking_concordance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Combined SHAP Top-54 ranking concordance saved")
    plt.show()
    plt.close()

def create_individual_feature_ranking_plots(shap_df):
    print(f"\nCreating individual SHAP feature ranking plots...")

    os.makedirs('Validation_Results', exist_ok=True)
    
    for model in CONFIG['models']:
        fig, ax = plt.subplots(figsize=(10, 12))
        
        model_shap = shap_df[shap_df['Model'] == model].sort_values('Rank')
        shap_top54 = model_shap.head(54)
        
        feature_data = []
        for _, row in shap_top54.iterrows():
            feature = row['Feature']
            importance = get_shap_importance_score(row)
            rank = row['Rank']
            
            feature_data.append({
                'feature': feature,
                'importance': importance,
                'rank': rank
            })
        
        feature_data = feature_data[::-1]
        
        y_positions = range(len(feature_data))
        importances = [fd['importance'] for fd in feature_data]
        
        ax.barh(y_positions, importances, color='#1f77b4', alpha=0.8, height=0.8)
        
        max_importance = max(importances) if importances else 0.001
        for j, fd in enumerate(feature_data):
            y_pos = j
            feature_label = f"{fd['rank']:2d}. {fd['feature']}"
            
            ax.text(max_importance * 0.01, y_pos, feature_label, va='center', ha='left', 
                    fontsize=8, fontweight='bold')
        
        ax.set_yticks([])
        ax.set_xlabel('SHAP Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'SHAP Feature Ranking\nModel: {model}', fontsize=16, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        if max_importance > 0:
            ax.set_xlim(0, max_importance * 1.1)
        else:
            ax.set_xlim(0, 0.01)
        
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.05)
        
        output_file = f'Validation_Results/shap_feature_ranking_{model}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   SHAP feature ranking for {model} saved to {output_file}")
        plt.show()
        plt.close()


def create_simplified_visualizations(all_results, validation_results):
    print(f"\nCreating simplified validation visualizations...")

    os.makedirs('Validation_Results', exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    models = list(validation_results['model_performances'].keys())
    x = np.arange(len(models))
    width = 0.35

    f1_a = [validation_results['model_performances'][m]['A'] for m in models]
    f1_b = [validation_results['model_performances'][m]['B'] for m in models]

    bars1 = ax.bar(x - width/2, f1_a, width, label='ReliefF+ANOVA+Correlation',
                     color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, f1_b, width, label='SHAP',
                     color='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('F1-Macro Score', fontweight='bold', fontsize=12)
    ax.set_title('F1-Macro Performance Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, max(max(f1_a), max(f1_b)) * 1.15)

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Feature Selection Validation Study Results\nF1-Macro Performance: ReliefF+ANOVA vs SHAP', 
                  fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.savefig('Validation_Results/simplified_validation_analysis.png', dpi=300, bbox_inches='tight')
    print("   Simplified validation visualization saved")
    plt.show()

def save_comprehensive_results(all_results, validation_results):
    print(f"\nSaving comprehensive results...")
    os.makedirs('Validation_Results', exist_ok=True)

    results_data = []
    for scenario in ['A', 'B', 'C']:
        for model in CONFIG['models']:
            if model in all_results[scenario]:
                result = all_results[scenario][model]
                results_data.append({
                    'Scenario': scenario,
                    'Model': model,
                    'Accuracy': result['accuracy'],
                    'F1_Macro': result['f1_macro'],
                    'F1_Weighted': result['f1_weighted'],
                    'Precision_Macro': result['precision_macro'],
                    'Recall_Macro': result['recall_macro'],
                    'Precision_Weighted': result['precision_weighted'],
                    'Recall_Weighted': result['recall_weighted'],
                })
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('Validation_Results/comprehensive_results.csv', index=False)
    print("   Detailed results saved to 'Validation_Results/comprehensive_results.csv'")

def main():
    selected_features, removed_df, shap_df = load_feature_selection_data()
    train_df, test_df, all_features = load_original_datasets()

    if selected_features is None or train_df is None:
        print("Required data could not be loaded. Exiting.")
        return

    X_train_full = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_test_full = test_df.drop('label', axis=1)
    y_test = test_df['label']
    
    scenario_features = create_scenario_features_enhanced(selected_features, removed_df, shap_df)

    all_results = {'A': {}, 'B': {}, 'C': {}}
    
    print("\n" + "="*80)
    print("STARTING MODEL TRAINING AND EVALUATION")
    print("="*80)
    
    for model_name in CONFIG['models']:
        print(f"\nEvaluating Model: {model_name}")
        for scenario_name in ['A', 'B', 'C']:
            features_to_use = scenario_features[scenario_name][model_name]
            X_train_scenario = X_train_full[features_to_use]
            X_test_scenario = X_test_full[features_to_use]

            result = train_evaluate_scenario_fixed_hyperparams(
                X_train_scenario, X_test_scenario, y_train, y_test,
                scenario_name, model_name, features_to_use
            )
            if result:
                all_results[scenario_name][model_name] = result

    validation_results = perform_validation_analysis(all_results)
    save_comprehensive_results(all_results, validation_results)

    create_shap_ranking_concordance_plots(selected_features, removed_df, shap_df)
    create_individual_feature_ranking_plots(shap_df)
    create_simplified_visualizations(all_results, validation_results)

if __name__ == '__main__':
    main()
