import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from itertools import combinations
import warnings
from collections import defaultdict

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# ====================================================================
#   RELIEFF + ANOVA + CORRELATION CONFIGURATION
# ====================================================================

CONFIG = {
    # ReliefF settings
    'relieff_k_neighbors': 15,         # Number of neighbors for ReliefF
    'relieff_max_iterations': 100,     # Maximum iterations for ReliefF
    'relieff_threshold': 0.01,         # Minimum threshold for ReliefF weights
    
    # ANOVA settings
    'anova_threshold': 1.0,            # Minimum F-score for ANOVA
    
    # Combination weights
    'relieff_weight': 0.5,             # Weight for ReliefF in final combination
    'anova_weight': 0.5,               # Weight for ANOVA in final combination
    
    # Correlation settings
    'correlation_threshold': 0.80,      # Threshold for redundancy (80%)
    
    # Selection settings
    'cumulative_threshold': 0.90,      # 90% of cumulative contribution
    'min_features': 5,                 # Minimum features to select
    'max_features': 144,               # Maximum features to select
    
    # General settings
    'verbose': True,
    'random_state': 42
}

# ====================================================================
# CONFIGURATION DISPLAY
# ====================================================================
def display_configuration():
    """Displays the current ReliefF + ANOVA configuration"""
    print(f"\n{'='*70}")
    print(" RELIEFF + ANOVA + CORRELATION - COMPLETE CONFIGURATION")
    print(f"{'='*70}")
    
    print(f" DATA SOURCE:")
    print(f"   Loading from: 'Preprocessed_Data/' folder")
    print(f"   Training file: dataset_train_transformed.csv")
    print(f"   Test file: dataset_test_transformed.csv")
    
    print(f"\n ALGORITHMS USED:")
    print(f"    Feature-Target Relevance: ReliefF (captures non-linear relationships)")
    print(f"    Feature-Target Relevance: ANOVA F-Score (mean differences)")
    print(f"    Feature-Feature Redundancy: Pearson Correlation")
    print(f"    Final selection: Cumulative contribution")
    
    print(f"\n RELIEFF PARAMETERS:")
    print(f"   K neighbors: {CONFIG['relieff_k_neighbors']}")
    print(f"   Max iterations: {CONFIG['relieff_max_iterations']}")
    print(f"   Minimum threshold: {CONFIG['relieff_threshold']}")
    
    print(f"\n ANOVA PARAMETERS:")
    print(f"   Minimum F-score: {CONFIG['anova_threshold']}")
    
    print(f"\n COMBINATION PARAMETERS:")
    print(f"   ReliefF weight: {CONFIG['relieff_weight']}")
    print(f"   ANOVA weight: {CONFIG['anova_weight']}")
    
    print(f"\n CORRELATION PARAMETERS:")
    print(f"   Redundancy threshold: {CONFIG['correlation_threshold']} (80%)")
    
    print(f"\n SELECTION BY CONTRIBUTION:")
    print(f"   Cumulative threshold: {CONFIG['cumulative_threshold']*100:.1f}%")
    print(f"   Min features: {CONFIG['min_features']}")
    print(f"   Max features: {CONFIG['max_features']}")
    
    return True

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def calculate_feature_redundancy(feature1_values, feature2_values):
    try:
        correlation, _ = pearsonr(feature1_values, feature2_values)
        redundancy = abs(correlation)
        if np.isnan(redundancy):
            return 0.0
        return redundancy
    except Exception as e:
        return 0.0

def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score - min_score == 0:
        return np.zeros_like(scores)
    
    return (scores - min_score) / (max_score - min_score)

# ====================================================================
# PLOTTING FUNCTIONS
# ====================================================================

def create_feature_selection_plots(selector, output_folder):
    """Create comprehensive plots for feature selection analysis"""
    
    # Create plots directory
    plots_dir = os.path.join(output_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\n CREATING VISUALIZATION PLOTS...")
    
    # 1. ReliefF vs ANOVA Scores Scatter Plot
    plot_relieff_vs_anova_scatter(selector, plots_dir)
    
    # 2. Top Features Ranking Bar Plot
    plot_top_features_ranking(selector, plots_dir)
    
    # 3. Feature Selection Process Overview
    plot_selection_process_overview(selector, plots_dir)
    
    # 4. Cumulative Contribution Plot
    plot_cumulative_contribution(selector, plots_dir)
    
    # 5. Score Distribution Histograms
    plot_score_distributions(selector, plots_dir)
    
    # 6. Correlation Heatmap (if available)
    if selector.correlation_matrix is not None:
        plot_correlation_heatmap(selector, plots_dir)
    
    # 7. Redundancy Analysis
    if selector.removed_features:
        plot_redundancy_analysis(selector, plots_dir)
    
    # 8. Feature Importance Comparison
    plot_feature_importance_comparison(selector, plots_dir)
    
    print(f" All plots saved in: {plots_dir}/")

def plot_relieff_vs_anova_scatter(selector, plots_dir):
    """Scatter plot comparing ReliefF vs ANOVA scores"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get all features data
    features = list(selector.relieff_scores.keys())
    relieff_scores = [selector.relieff_scores[f] for f in features]
    anova_scores = [selector.anova_scores[f] for f in features]
    
    # Color based on selection status
    colors = ['red' if f in selector.selected_features else 'lightblue' for f in features]
    sizes = [50 if f in selector.selected_features else 20 for f in features]
    
    scatter = ax.scatter(relieff_scores, anova_scores, c=colors, s=sizes, alpha=0.6)
    
    # Add threshold lines
    ax.axvline(x=selector.relieff_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'ReliefF Threshold ({selector.relieff_threshold})')
    ax.axhline(y=selector.anova_threshold, color='blue', linestyle='--', alpha=0.7, 
               label=f'ANOVA Threshold ({selector.anova_threshold})')
    
    ax.set_xlabel('ReliefF Score', fontsize=12)
    ax.set_ylabel('ANOVA F-Score', fontsize=12)
    ax.set_title('Feature Selection: ReliefF vs ANOVA Scores', fontsize=14, fontweight='bold')
    
    # Create custom legend
    selected_patch = mpatches.Patch(color='red', label=f'Selected Features ({len(selector.selected_features)})')
    not_selected_patch = mpatches.Patch(color='lightblue', label=f'Not Selected ({len(features) - len(selector.selected_features)})')
    ax.legend(handles=[selected_patch, not_selected_patch], loc='upper right')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'relieff_vs_anova_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_features_ranking(selector, plots_dir, top_n=20):
    """Bar plot of top N selected features"""
    if not selector.selected_features:
        return
    
    # Get top features sorted by combined score
    selected_with_scores = [(f, selector.combined_scores.get(f, 0)) for f in selector.selected_features]
    selected_sorted = sorted(selected_with_scores, key=lambda x: x[1], reverse=True)
    
    top_features = selected_sorted[:top_n]
    features_names = [f[0] for f in top_features]
    combined_scores = [f[1] for f in top_features]
    
    # Get individual scores for stacking
    relieff_scores = [selector.relieff_scores.get(f, 0) for f, _ in top_features]
    anova_scores = [selector.anova_scores.get(f, 0) for f, _ in top_features]
    
    # Normalize for visualization
    relieff_norm = normalize_scores(relieff_scores)
    anova_norm = normalize_scores(anova_scores)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Top plot: Combined scores
    bars1 = ax1.barh(range(len(features_names)), combined_scores, color='steelblue', alpha=0.8)
    ax1.set_yticks(range(len(features_names)))
    ax1.set_yticklabels([f[:30] + '...' if len(f) > 30 else f for f in features_names], fontsize=10)
    ax1.set_xlabel('Combined Score (ReliefF + ANOVA)', fontsize=12)
    ax1.set_title(f'Top {len(top_features)} Selected Features - Combined Scores', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars1, combined_scores)):
        ax1.text(score + 0.001, i, f'{score:.3f}', va='center', fontsize=9)
    
    # Bottom plot: Individual score comparison
    x_pos = np.arange(len(features_names))
    width = 0.35
    
    bars2 = ax2.barh(x_pos - width/2, relieff_norm, width, label='ReliefF (normalized)', 
                     color='orangered', alpha=0.7)
    bars3 = ax2.barh(x_pos + width/2, anova_norm, width, label='ANOVA (normalized)', 
                     color='forestgreen', alpha=0.7)
    
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels([f[:30] + '...' if len(f) > 30 else f for f in features_names], fontsize=10)
    ax2.set_xlabel('Normalized Scores', fontsize=12)
    ax2.set_title('Individual Algorithm Scores Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'top_features_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_selection_process_overview(selector, plots_dir):
    """Overview of the feature selection process steps"""
    
    # Calculate step-by-step feature counts
    total_features = len(selector.relieff_scores)
    relieff_relevant = len([f for f, s in selector.relieff_scores.items() if s >= selector.relieff_threshold])
    anova_relevant = len([f for f, s in selector.anova_scores.items() if s >= selector.anova_threshold])
    
    # Union of relevant features
    relieff_features = set([f for f, s in selector.relieff_scores.items() if s >= selector.relieff_threshold])
    anova_features = set([f for f, s in selector.anova_scores.items() if s >= selector.anova_threshold])
    combined_relevant = len(relieff_features.union(anova_features))
    
    # After redundancy removal
    after_redundancy = combined_relevant - len([f for f in selector.removed_features.values() if f['reason'] == 'Redundant'])
    final_selected = len(selector.selected_features)
    
    steps = ['Original\nFeatures', 'ReliefF\nRelevant', 'ANOVA\nRelevant', 'Combined\nRelevant', 
             'After\nRedundancy\nRemoval', 'Final\nSelected']
    counts = [total_features, relieff_relevant, anova_relevant, combined_relevant, after_redundancy, final_selected]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bar chart
    bars = ax.bar(steps, counts, color=['lightgray', 'orange', 'green', 'blue', 'purple', 'red'], 
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add reduction percentages
    for i in range(1, len(counts)):
        reduction = ((counts[0] - counts[i]) / counts[0]) * 100
        ax.text(i, counts[i] - max(counts)*0.05, f'-{reduction:.1f}%', 
                ha='center', va='top', fontsize=10, color='white', fontweight='bold')
    
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Feature Selection Process Overview', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add process flow arrows
    for i in range(len(steps)-1):
        ax.annotate('', xy=(i+1, max(counts)*0.9), xytext=(i, max(counts)*0.9),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'selection_process_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cumulative_contribution(selector, plots_dir):
    """Plot cumulative contribution of selected features"""
    if not selector.feature_contributions:
        return
    
    # Sort features by contribution
    sorted_contributions = sorted(selector.feature_contributions.items(), key=lambda x: x[1], reverse=True)
    
    features = [f for f, _ in sorted_contributions]
    contributions = [c for _, c in sorted_contributions]
    cumulative_contributions = np.cumsum(contributions)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Top plot: Individual contributions
    bars = ax1.bar(range(len(features)), contributions, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Feature Rank', fontsize=12)
    ax1.set_ylabel('Individual Contribution (%)', fontsize=12)
    ax1.set_title('Individual Feature Contributions', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Highlight top contributors
    for i, (bar, contrib) in enumerate(zip(bars[:10], contributions[:10])):
        ax1.text(i, contrib + max(contributions)*0.01, f'{contrib:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    # Bottom plot: Cumulative contribution
    ax2.plot(range(len(features)), cumulative_contributions, 'o-', color='red', linewidth=2, markersize=4)
    ax2.axhline(y=selector.cumulative_threshold*100, color='green', linestyle='--', 
                label=f'Threshold ({selector.cumulative_threshold*100:.0f}%)')
    ax2.fill_between(range(len(features)), cumulative_contributions, alpha=0.3, color='red')
    
    ax2.set_xlabel('Number of Features', fontsize=12)
    ax2.set_ylabel('Cumulative Contribution (%)', fontsize=12)
    ax2.set_title('Cumulative Feature Contribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Mark the cutoff point
    cutoff_idx = len([c for c in cumulative_contributions if c <= selector.cumulative_threshold*100])
    if cutoff_idx < len(cumulative_contributions):
        ax2.axvline(x=cutoff_idx, color='green', linestyle=':', alpha=0.7)
        ax2.text(cutoff_idx, 50, f'Cutoff at\n{cutoff_idx+1} features', 
                ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cumulative_contribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_distributions(selector, plots_dir):
    """Plot distributions of ReliefF and ANOVA scores"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # ReliefF score distribution
    relieff_scores = list(selector.relieff_scores.values())
    selected_relieff = [selector.relieff_scores[f] for f in selector.selected_features]
    
    ax1.hist(relieff_scores, bins=50, alpha=0.7, color='orange', label='All Features', density=True)
    ax1.hist(selected_relieff, bins=30, alpha=0.8, color='red', label='Selected Features', density=True)
    ax1.axvline(x=selector.relieff_threshold, color='black', linestyle='--', 
                label=f'Threshold ({selector.relieff_threshold})')
    ax1.set_xlabel('ReliefF Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('ReliefF Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # ANOVA score distribution
    anova_scores = list(selector.anova_scores.values())
    selected_anova = [selector.anova_scores[f] for f in selector.selected_features]
    
    ax2.hist(anova_scores, bins=50, alpha=0.7, color='green', label='All Features', density=True)
    ax2.hist(selected_anova, bins=30, alpha=0.8, color='red', label='Selected Features', density=True)
    ax2.axvline(x=selector.anova_threshold, color='black', linestyle='--', 
                label=f'Threshold ({selector.anova_threshold})')
    ax2.set_xlabel('ANOVA F-Score', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('ANOVA Score Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Combined score distribution
    combined_scores = list(selector.combined_scores.values())
    selected_combined = [selector.combined_scores[f] for f in selector.selected_features]
    
    ax3.hist(combined_scores, bins=50, alpha=0.7, color='blue', label='All Features', density=True)
    ax3.hist(selected_combined, bins=30, alpha=0.8, color='red', label='Selected Features', density=True)
    ax3.set_xlabel('Combined Score', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Combined Score Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Box plot comparison
    data_to_plot = [relieff_scores, anova_scores, combined_scores]
    selected_data = [selected_relieff, selected_anova, selected_combined]
    
    bp1 = ax4.boxplot(data_to_plot, positions=[1, 2, 3], widths=0.3, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
    bp2 = ax4.boxplot(selected_data, positions=[1.4, 2.4, 3.4], widths=0.3, patch_artist=True,
                      boxprops=dict(facecolor='red', alpha=0.7))
    
    # Fix the x-axis labels and ticks for box plots
    ax4.set_xticks([1.2, 2.2, 3.2])  # Set positions between the paired box plots
    ax4.set_xticklabels(['ReliefF', 'ANOVA', 'Combined'])
    ax4.set_ylabel('Score Value', fontsize=12)
    ax4.set_title('Score Distributions Comparison', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add legend for box plots
    all_patch = mpatches.Patch(color='lightblue', alpha=0.7, label='All Features')
    selected_patch = mpatches.Patch(color='red', alpha=0.7, label='Selected Features')
    ax4.legend(handles=[all_patch, selected_patch])
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'score_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(selector, plots_dir, max_features=50):
    """Plot correlation heatmap of selected features"""
    if selector.correlation_matrix is None or len(selector.selected_features) == 0:
        return
    
    # Get relevant features for correlation matrix
    relevant_relieff = set([f for f, s in selector.relieff_scores.items() if s >= selector.relieff_threshold])
    relevant_anova = set([f for f, s in selector.anova_scores.items() if s >= selector.anova_threshold])
    relevant_combined = list(relevant_relieff.union(relevant_anova))
    
    # Limit to top features if too many
    if len(relevant_combined) > max_features:
        # Sort by combined score and take top features
        relevant_with_scores = [(f, selector.combined_scores.get(f, 0)) for f in relevant_combined]
        relevant_sorted = sorted(relevant_with_scores, key=lambda x: x[1], reverse=True)
        relevant_combined = [f for f, _ in relevant_sorted[:max_features]]
    
    correlation_matrix = selector.correlation_matrix
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.1, cbar_kws={"shrink": .8}, ax=ax)
    
    # Highlight high correlations
    high_corr_indices = np.where(np.abs(correlation_matrix) >= selector.correlation_threshold)
    for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
        if i != j:  # Don't highlight diagonal
            rect = Rectangle((j, i), 1, 1, linewidth=3, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)
    
    ax.set_title(f'Feature Correlation Matrix\n(Relevant Features, Yellow boxes: |r| ≥ {selector.correlation_threshold})', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_redundancy_analysis(selector, plots_dir):
    """Plot analysis of removed redundant features"""
    redundant_features = [f for f, info in selector.removed_features.items() if info['reason'] == 'Redundant']
    
    if not redundant_features:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Redundant vs Selected features scores
    redundant_relieff = [selector.relieff_scores.get(f, 0) for f in redundant_features]
    redundant_anova = [selector.anova_scores.get(f, 0) for f in redundant_features]
    redundant_combined = [selector.combined_scores.get(f, 0) for f in redundant_features]
    
    selected_relieff = [selector.relieff_scores.get(f, 0) for f in selector.selected_features]
    selected_anova = [selector.anova_scores.get(f, 0) for f in selector.selected_features]
    selected_combined = [selector.combined_scores.get(f, 0) for f in selector.selected_features]
    
    # Box plot comparison
    data_redundant = [redundant_relieff, redundant_anova, redundant_combined]
    data_selected = [selected_relieff, selected_anova, selected_combined]
    
    bp1 = ax1.boxplot(data_redundant, positions=[1, 2, 3], widths=0.3, patch_artist=True,
                     boxprops=dict(facecolor='red', alpha=0.7), labels=['ReliefF', 'ANOVA', 'Combined'])
    bp2 = ax1.boxplot(data_selected, positions=[1.4, 2.4, 3.4], widths=0.3, patch_artist=True,
                     boxprops=dict(facecolor='green', alpha=0.7))
    
    ax1.set_ylabel('Score Value', fontsize=12)
    ax1.set_title('Score Comparison: Redundant vs Selected Features', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add legend
    redundant_patch = mpatches.Patch(color='red', alpha=0.7, label=f'Redundant ({len(redundant_features)})')
    selected_patch = mpatches.Patch(color='green', alpha=0.7, label=f'Selected ({len(selector.selected_features)})')
    ax1.legend(handles=[redundant_patch, selected_patch])
    
    # Right plot: Correlation distribution of removed features
    correlations = []
    for f, info in selector.removed_features.items():
        if info['reason'] == 'Redundant':
            correlations.append(abs(info['correlation']))
    
    ax2.hist(correlations, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(x=selector.correlation_threshold, color='black', linestyle='--', 
                label=f'Threshold ({selector.correlation_threshold})')
    ax2.set_xlabel('Absolute Correlation Coefficient', fontsize=12)
    ax2.set_ylabel('Number of Removed Features', fontsize=12)
    ax2.set_title('Distribution of Correlations\n(Removed Features)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'redundancy_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance_comparison(selector, plots_dir, top_n=54):
    """Compare feature importance across different methods"""
    if not selector.selected_features:
        return
    
    # Get top features
    selected_with_scores = [(f, selector.combined_scores.get(f, 0)) for f in selector.selected_features]
    selected_sorted = sorted(selected_with_scores, key=lambda x: x[1], reverse=True)
    top_features = selected_sorted[:top_n]
    
    features_names = [f[0] for f in top_features]
    
    # Get scores for each method
    relieff_scores = [selector.relieff_scores.get(f, 0) for f, _ in top_features]
    anova_scores = [selector.anova_scores.get(f, 0) for f, _ in top_features]
    combined_scores = [f[1] for f in top_features]
    contributions = [selector.feature_contributions.get(f, 0) for f, _ in top_features]
    
    # Normalize all scores for comparison
    relieff_norm = normalize_scores(relieff_scores)
    anova_norm = normalize_scores(anova_scores)
    combined_norm = normalize_scores(combined_scores)
    contrib_norm = normalize_scores(contributions)
    
    # Create radar chart style comparison
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(features_names))
    width = 0.2
    
    bars1 = ax.bar(x - width*1.5, relieff_norm, width, label='ReliefF (norm)', alpha=0.8, color='orange')
    bars2 = ax.bar(x - width*0.5, anova_norm, width, label='ANOVA (norm)', alpha=0.8, color='green')
    bars3 = ax.bar(x + width*0.5, combined_norm, width, label='Combined (norm)', alpha=0.8, color='blue')
    bars4 = ax.bar(x + width*1.5, contrib_norm, width, label='Contribution (norm)', alpha=0.8, color='red')
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title(f'Feature Importance Comparison - Top {len(top_features)} Features', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f[:20] + '...' if len(f) > 20 else f for f in features_names], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ====================================================================
# ReliefFAnovaCorrelationSelector CLASS
# ====================================================================

class ReliefFAnovaCorrelationSelector:
    def __init__(self, **config):
        self.selected_features = []
        self.relieff_scores = {}
        self.anova_scores = {}
        self.combined_scores = {}
        self.removed_features = {}
        self.correlation_matrix = None
        self.feature_contributions = {}
        
        # Apply CONFIG settings
        for key, value in CONFIG.items():
            setattr(self, key, value)
        
        # Override with provided parameters
        for key, value in config.items():
            setattr(self, key, value)
    
    def _calculate_relieff_weights(self, X, y):
        """Implementation of the ReliefF algorithm"""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        
        # Label encoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        classes = np.unique(y_encoded)
        
        # ReliefF iterations
        iterations = min(self.relieff_max_iterations, n_samples)
        
        for iteration in range(iterations):
            # Select random instance
            np.random.seed(self.random_state + iteration)
            instance_idx = np.random.randint(0, n_samples)
            instance = X[instance_idx]
            instance_class = y_encoded[instance_idx]
            
            # Compute distances to all other instances
            distances = np.sqrt(np.sum((X - instance) ** 2, axis=1))
            
            # Find nearest hits (same class)
            same_class_mask = (y_encoded == instance_class) & (np.arange(n_samples) != instance_idx)
            same_class_distances = distances[same_class_mask]
            
            nearest_hits_indices = []
            if len(same_class_distances) > 0:
                same_class_indices = np.where(same_class_mask)[0]
                k_hits = min(self.relieff_k_neighbors, len(same_class_indices))
                nearest_hits_indices = same_class_indices[np.argsort(same_class_distances)[:k_hits]]
            
            # Find nearest misses (different classes)
            nearest_misses_indices = []
            for other_class in classes:
                if other_class != instance_class:
                    other_class_mask = y_encoded == other_class
                    other_class_distances = distances[other_class_mask]
                    
                    if len(other_class_distances) > 0:
                        other_class_indices = np.where(other_class_mask)[0]
                        k_miss = min(self.relieff_k_neighbors, len(other_class_indices))
                        nearest_miss_class = other_class_indices[np.argsort(other_class_distances)[:k_miss]]
                        nearest_misses_indices.extend(nearest_miss_class)
            
            # Update feature weights
            for feature_idx in range(n_features):
                # Difference with nearest hits (penalize)
                hit_diff = 0
                if len(nearest_hits_indices) > 0:
                    for hit_idx in nearest_hits_indices:
                        hit_diff += abs(instance[feature_idx] - X[hit_idx][feature_idx])
                    hit_diff /= len(nearest_hits_indices)
                
                # Difference with nearest misses (reward)
                miss_diff = 0
                if len(nearest_misses_indices) > 0:
                    for miss_idx in nearest_misses_indices:
                        miss_diff += abs(instance[feature_idx] - X[miss_idx][feature_idx])
                    miss_diff /= len(nearest_misses_indices)
                
                # Update weight: increment if miss_diff > hit_diff
                weights[feature_idx] += (miss_diff - hit_diff) / iterations
        
        return weights
    
    def _calculate_anova_fscore(self, X, y):
        """Calculate ANOVA F-score for each feature"""
        f_scores = []
        
        for i in range(X.shape[1]):
            feature_values = X[:, i]
            
            # Group by class
            classes = np.unique(y)
            groups = [feature_values[y == cls] for cls in classes]
            
            try:
                # Compute F-statistic
                f_stat, p_value = stats.f_oneway(*groups)
                f_scores.append(f_stat if not np.isnan(f_stat) else 0)
            except:
                f_scores.append(0)
        
        return np.array(f_scores)
    
    def _calculate_correlation_matrix(self, X):
        """Calculate correlation matrix between features"""
        try:
            correlation_matrix = np.corrcoef(X.T)
            return correlation_matrix
        except:
            return np.eye(X.shape[1])  # Identity matrix as fallback
    
    def _find_redundant_features(self, correlation_matrix, feature_names):
        """Find pairs of features with high correlation"""
        redundant_pairs = []
        n_features = len(feature_names)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(correlation_matrix[i, j]) >= self.correlation_threshold:
                    redundant_pairs.append((i, j, correlation_matrix[i, j], feature_names[i], feature_names[j]))
        
        return redundant_pairs
    
    def _remove_redundant_features(self, feature_scores, feature_names, redundant_pairs):
        """Remove redundant features keeping the one with the highest score"""
        features_to_remove = set()
        
        for i, j, corr, name_i, name_j in redundant_pairs:
            # Keep feature with higher combined score
            score_i = feature_scores.get(name_i, 0)
            score_j = feature_scores.get(name_j, 0)
            
            if score_i > score_j:
                features_to_remove.add(name_j)
                self.removed_features[name_j] = {
                    'reason': 'Redundant',
                    'correlated_with': name_i,
                    'correlation': corr,
                    'combined_score': score_j
                }
            else:
                features_to_remove.add(name_i)
                self.removed_features[name_i] = {
                    'reason': 'Redundant',
                    'correlated_with': name_j,
                    'correlation': corr,
                    'combined_score': score_i
                }
        
        return list(features_to_remove)
    
    def _calculate_feature_contributions(self, final_scores):
        """Calculate percentage contribution of each feature"""
        total_score = sum(final_scores.values())
        
        if total_score == 0:
            return {}
        
        contributions = {}
        for feature, score in final_scores.items():
            contributions[feature] = (score / total_score) * 100
        
        return contributions
    
    def _apply_cumulative_selection(self, feature_scores, verbose=True):
        """Apply selection based on cumulative contribution"""
        # Sort features by score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        total_score = sum(feature_scores.values())
        if total_score == 0:
            if verbose:
                print("   Total score is zero. No feature selected.")
            return []
        
        cumulative_score = 0
        selected_features = []
        
        for feature, score in sorted_features:
            cumulative_score += score
            selected_features.append(feature)
            
            cumulative_percentage = cumulative_score / total_score
            
            # Stop if reaching cumulative threshold
            if cumulative_percentage >= self.cumulative_threshold:
                break
            
            # Stop if reaching maximum number of features
            if len(selected_features) >= self.max_features:
                break
        
        # Ensure minimum number of features
        if len(selected_features) < self.min_features:
            remaining_features = [f for f, s in sorted_features if f not in selected_features]
            needed = self.min_features - len(selected_features)
            selected_features.extend(remaining_features[:needed])
        
        return selected_features
    
    def fit_transform(self, X_train_raw, y_train, X_test_raw, y_test, verbose=None):
        """
        Main feature selection process:
        1. RELIEFF: Identifies relevant features (non-linear relationships)
        2. ANOVA: Identifies relevant features (mean differences)
        3. COMBINATION: Combines ReliefF + ANOVA scores
        4. CORRELATION: Removes redundancy
        5. CUMULATIVE CONTRIBUTION: Final selection
        """
        
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"   ADVANCED SELECTION: RELIEFF + ANOVA + CORRELATION")
            print(f"{'='*70}")
            print(f"   Input: {X_train_raw.shape[0]} samples, {X_train_raw.shape[1]} features")
        
        # Copy data
        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()
        feature_names = list(X_train.columns)
        
        # Convert to numpy
        X_train_array = X_train.values
        X_test_array = X_test.values
        
        # === STEP 1: RELIEFF WEIGHTS ===
        if verbose:
            print(f"\nSTEP 1: RELIEFF - Calculating weights")
        
        relieff_weights = self._calculate_relieff_weights(X_train_array, y_train)
        
        # Store ReliefF scores
        for i, feature in enumerate(feature_names):
            self.relieff_scores[feature] = relieff_weights[i]
        
        # Filter relevant features by ReliefF
        relevant_relieff = [f for f in feature_names if self.relieff_scores[f] >= self.relieff_threshold]
        
        if verbose:
            print(f"   Relevant features by ReliefF (>= {self.relieff_threshold}): {len(relevant_relieff)}")
        
        # === STEP 2: ANOVA F-SCORE ===
        if verbose:
            print(f"\nSTEP 2: ANOVA - Calculating F-scores")
        
        anova_fscores = self._calculate_anova_fscore(X_train_array, y_train)
        
        # Store ANOVA scores
        for i, feature in enumerate(feature_names):
            self.anova_scores[feature] = anova_fscores[i]
        
        # Filter relevant features by ANOVA
        relevant_anova = [f for f in feature_names if self.anova_scores[f] >= self.anova_threshold]
        
        if verbose:
            print(f"   Relevant features by ANOVA (>= {self.anova_threshold}): {len(relevant_anova)}")
        
        # === STEP 3: COMBINATION RELIEFF + ANOVA ===
        if verbose:
            print(f"\nSTEP 3: COMBINATION - ReliefF + ANOVA")
        
        # Normalize scores
        relieff_normalized = normalize_scores(list(self.relieff_scores.values()))
        anova_normalized = normalize_scores(list(self.anova_scores.values()))
        
        # Combine scores
        for i, feature in enumerate(feature_names):
            combined_score = (self.relieff_weight * relieff_normalized[i] + 
                            self.anova_weight * anova_normalized[i])
            self.combined_scores[feature] = combined_score
        
        # Filter features relevant by at least one method
        relevant_combined = list(set(relevant_relieff + relevant_anova))
        relevant_combined_scores = {f: self.combined_scores[f] for f in relevant_combined}
        
        if verbose:
            print(f"   Relevant features (ReliefF OR ANOVA): {len(relevant_combined)}")
            print(f"   ReliefF weight: {self.relieff_weight}, ANOVA weight: {self.anova_weight}")
        
        # === STEP 4: CORRELATION (Redundancy Removal) ===
        if verbose:
            print(f"\nSTEP 4: CORRELATION - Removing redundancy")
        
        if len(relevant_combined) > 1:
            # Compute correlation matrix only for relevant features
            relevant_indices = [feature_names.index(f) for f in relevant_combined]
            X_relevant = X_train_array[:, relevant_indices]
            
            correlation_matrix = self._calculate_correlation_matrix(X_relevant)
            self.correlation_matrix = correlation_matrix
            
            # Find redundant pairs
            redundant_pairs = self._find_redundant_features(correlation_matrix, relevant_combined)
            
            if verbose:
                print(f"   Pairs with correlation >= {self.correlation_threshold}: {len(redundant_pairs)}")
            
            # Remove redundant features
            features_to_remove = self._remove_redundant_features(
                relevant_combined_scores, relevant_combined, redundant_pairs
            )
            
            # Final features after redundancy removal
            features_after_redundancy = [f for f in relevant_combined if f not in features_to_remove]
            final_scores = {f: relevant_combined_scores[f] for f in features_after_redundancy}
            
            if verbose:
                print(f"   Features removed by redundancy: {len(features_to_remove)}")
                print(f"   Remaining features: {len(features_after_redundancy)}")
        else:
            final_scores = relevant_combined_scores
            features_after_redundancy = relevant_combined
        
        # === STEP 5: CUMULATIVE CONTRIBUTION (Final Selection) ===
        if verbose:
            print(f"\nSTEP 5: CUMULATIVE CONTRIBUTION - Final selection")
        
        # Compute contributions
        self.feature_contributions = self._calculate_feature_contributions(final_scores)
        
        # Apply cumulative selection
        self.selected_features = self._apply_cumulative_selection(final_scores, verbose)
        
        if verbose:
            print(f"   Cumulative threshold: {self.cumulative_threshold*100:.1f}%")
            print(f"   Selected features: {len(self.selected_features)}")
            
            # Compute total contribution of selected features
            total_contribution = sum(self.feature_contributions.get(f, 0) for f in self.selected_features)
            print(f"   Total captured contribution: {total_contribution:.1f}%")
        
        # Final logging
        if verbose:
            print(f"\nSELECTION COMPLETE:")
            print(f"   Original features: {X_train_raw.shape[1]}")
            print(f"   Selected features: {len(self.selected_features)}")
            reduction_pct = ((X_train_raw.shape[1] - len(self.selected_features)) / X_train_raw.shape[1] * 100)
            print(f"   Total reduction: {reduction_pct:.1f}%")
        
        # Return filtered DataFrames
        return X_train[self.selected_features], X_test[self.selected_features]
    
    def get_feature_report(self):
        """Generate comprehensive feature selection report"""
        if not self.selected_features:
            return "No feature selected yet."
        
        report = "\n=== REPORT: RELIEFF + ANOVA + CORRELATION ===\n"
        report += f"{'='*70}\n"
        report += "FINAL SELECTED FEATURES\n"
        report += f"{'='*70}\n"
        report += f"{'Rank':<4} {'Feature':<45} {'ReliefF':<10} {'ANOVA':<10} {'Combined':<10} {'Contrib%':<8}\n"
        report += "-" * 85 + "\n"
        
        # Sort selected features by combined score
        selected_with_scores = [(f, self.combined_scores.get(f, 0)) for f in self.selected_features]
        selected_sorted = sorted(selected_with_scores, key=lambda x: x[1], reverse=True)
        
        for i, (feature, _) in enumerate(selected_sorted, 1):
            relieff_score = self.relieff_scores.get(feature, 0)
            anova_score = self.anova_scores.get(feature, 0)
            combined_score = self.combined_scores.get(feature, 0)
            contribution = self.feature_contributions.get(feature, 0)
            
            report += f"{i:<4} {feature[:44]:<45} {relieff_score:<10.4f} {anova_score:<10.4f} {combined_score:<10.4f} {contribution:<8.2f}\n"
        
        # Removal statistics
        if self.removed_features:
            report += f"\n{'='*70}\n"
            report += "REMOVAL STATISTICS\n"
            report += f"{'='*70}\n"
            
            removal_summary = defaultdict(int)
            for info in self.removed_features.values():
                removal_summary[info['reason']] += 1
            
            for reason, count in removal_summary.items():
                report += f"   {reason}: {count} features\n"
            
            report += f"\nTotal removed features: {len(self.removed_features)}\n"
        
        return report
    
    def get_top_features_summary(self, n=20):
        """Return summary of top N features"""
        if not self.selected_features:
            return "No feature selected yet."
        
        summary = f"\nTOP {min(n, len(self.selected_features))} SELECTED FEATURES:\n"
        summary += "="*60 + "\n"
        
        # Sort by combined score
        selected_with_scores = [(f, self.combined_scores.get(f, 0)) for f in self.selected_features]
        selected_sorted = sorted(selected_with_scores, key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(selected_sorted[:n], 1):
            contribution = self.feature_contributions.get(feature, 0)
            summary += f"{i:2d}. {feature:<40} (Score: {score:.4f}, Contrib: {contribution:.1f}%)\n"
        
        return summary


# ====================================================================
# UTILITY FUNCTIONS (Validation and Saving)
# ====================================================================

def validate_data_integrity(X_data, y_data, stage_name, verbose=True):
    """Validate if X and y are aligned and without issues"""
    if verbose:
        print(f"\nVALIDATING: {stage_name}")
    
    issues = []
    
    if X_data.empty:
        issues.append("X_data is empty")
    if len(y_data) == 0:
        issues.append("y_data is empty")
    
    if len(X_data) != len(y_data):
        issues.append(f"Sizes do not match: X={len(X_data)}, y={len(y_data)}")
    
    if hasattr(y_data, 'isna'):
        missing_labels = y_data.isna().sum()
        if missing_labels > 0:
            issues.append(f"{missing_labels} missing labels")
    
    if not X_data.empty:
        total_nans = X_data.isna().sum().sum()
        if total_nans > 0 and verbose:
            print(f"   {total_nans} NaNs in features (ok if treated)")
    
    if issues:
        print(f"   ISSUES FOUND:")
        for issue in issues:
            print(f"      - {issue}")
        return False
    else:
        if verbose:
            print(f"   Data integrity OK: {len(X_data)} samples")
        return True

def safe_save_dataset(X_data, y_data, filepath, stage_name):
    """Save dataset with full integrity validation"""
    print(f"\nSAVING: {stage_name}")
    print(f"   File: {os.path.basename(filepath)}")
    
    if not validate_data_integrity(X_data, y_data, f"{stage_name} (pre-save)", verbose=False):
            print(f"   FAILED: Invalid data for {stage_name}")
            return False

    try:
            X_aligned = X_data.copy().reset_index(drop=True)
            y_aligned = pd.Series(y_data).reset_index(drop=True)

            if len(X_aligned) != len(y_aligned):
                print(f"   ERROR: Sizes still do not match: X={len(X_aligned)}, y={len(y_aligned)}")
                return False

            dataset_final = X_aligned.copy()
            dataset_final['label'] = y_aligned

            dataset_final.to_csv(filepath, index=False)

            # Verification
            df_verify = pd.read_csv(filepath)
            print(f"   SUCCESS: {df_verify.shape[0]} rows, {df_verify.shape[1]-1} features")
            print(f"   Classes: {df_verify['label'].unique().tolist()}")
            return True

    except Exception as e:
            print(f"   ERROR saving file: {e}")
            return False


# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("ADVANCED FEATURE SELECTION: RELIEFF + ANOVA + CORRELATION")
    print("="*75)

    # Show configuration
    display_configuration()

    # Load preprocessed datasets
    print("\nLOADING PREPROCESSED DATASETS...")
    PREPROCESSED_FOLDER = 'Preprocessed_Data'
    train_path = os.path.join(PREPROCESSED_FOLDER, 'dataset_train_transformed.csv')
    test_path = os.path.join(PREPROCESSED_FOLDER, 'dataset_test_transformed.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"CRITICAL ERROR: Train/test datasets not found in '{PREPROCESSED_FOLDER}/'")
        print("   Make sure you have run the preprocessing pipeline first.")
        return

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Dataset train loaded: {df_train.shape}")
    print(f"Dataset test loaded: {df_test.shape}")

    # Prepare features and target
    feature_columns = [col for col in df_train.columns if col != 'label']
    X_train_raw = df_train[feature_columns].copy()
    y_train_original = df_train['label'].copy()
    X_test_raw = df_test[feature_columns].copy()
    y_test_original = df_test['label'].copy()

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_original)
    y_test_encoded = le.transform(y_test_original)

    print("\nData loaded and labels encoded.")
    print(f"   Classes: {le.classes_.tolist()}")

    # Cleaning and scaling
    print("\nCLEANING AND STANDARDIZING DATA...")
    X_train_clean = X_train_raw.replace([np.inf, -np.inf], np.nan).fillna(X_train_raw.median(numeric_only=True))
    X_test_clean = X_test_raw.replace([np.inf, -np.inf], np.nan).fillna(X_train_raw.median(numeric_only=True))

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_clean),
        columns=X_train_clean.columns,
        index=X_train_clean.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_clean),
        columns=X_test_clean.columns,
        index=X_test_clean.index
    )
    print("Data cleaned and standardized. Ready for selection.")

    # Instantiate selector
    selector = ReliefFAnovaCorrelationSelector(**CONFIG)

    # Apply selection
    X_train_final, X_test_final = selector.fit_transform(
        X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, verbose=True
    )

    # Show results
    print("\n" + "="*70)
    print("FINAL SELECTION RESULTS:")
    print("="*70)
    print(f"   Original features: {X_train_raw.shape[1]}")
    print(f"   Selected features: {len(selector.selected_features)}")
    reduction_pct = ((X_train_raw.shape[1] - len(selector.selected_features)) / X_train_raw.shape[1] * 100)
    print(f"   Total reduction: {reduction_pct:.1f}%")
    print(f"   Features removed due to redundancy: {len([f for f in selector.removed_features.values() if f['reason'] == 'Redundant'])}")

    # Show top features
    print(selector.get_top_features_summary(54))

    # Setup output directory
    output_folder = 'ReliefF_ANOVA_Correlation_Results'
    os.makedirs(output_folder, exist_ok=True)

    # ====================================================================
    # CREATE VISUALIZATION PLOTS
    # ====================================================================

    create_feature_selection_plots(selector, output_folder)

    # ====================================================================
    # SAVE RESULTS
    # ====================================================================

    # Save selected features with details
    try:
        features_data = []
        selected_sorted = sorted(
            [(f, selector.combined_scores.get(f, 0)) for f in selector.selected_features],
            key=lambda x: x[1], reverse=True
        )

        for i, (feature, _) in enumerate(selected_sorted, 1):
            features_data.append({
                'Rank_Final': i,
                'Feature': feature,
                'ReliefF_Score': selector.relieff_scores.get(feature, np.nan),
                'ANOVA_Score': selector.anova_scores.get(feature, np.nan),
                'Combined_Score': selector.combined_scores.get(feature, np.nan),
                'Contribution_Percent': selector.feature_contributions.get(feature, np.nan),
                'Removed': False
            })

        features_df = pd.DataFrame(features_data)
        features_path = os.path.join(output_folder, 'selected_features_detailed_ranking.csv')
        features_df.to_csv(features_path, index=False)
        print(f"Detailed ranking saved: {os.path.basename(features_path)}")
    except Exception as e:
        print(f"Error saving ranking: {e}")

    # Save removed features
    if selector.removed_features:
        try:
            removed_df = pd.DataFrame.from_dict(selector.removed_features, orient='index')
            removed_df.index.name = 'Feature'
            removed_path = os.path.join(output_folder, 'removed_features_redundancy.csv')
            removed_df.to_csv(removed_path)
            print(f"Removed features saved: {os.path.basename(removed_path)}")
        except Exception as e:
            print(f"Error saving removed features: {e}")

    # Save filtered datasets
    try:
        train_filtered = X_train_final.copy()
        train_filtered['label'] = y_train_original.reset_index(drop=True)
        train_path_filtered = os.path.join(output_folder, 'dataset_train_ReliefF_ANOVA_filtered.csv')
        train_filtered.to_csv(train_path_filtered, index=False)

        test_filtered = X_test_final.copy()
        test_filtered['label'] = y_test_original.reset_index(drop=True)
        test_path_filtered = os.path.join(output_folder, 'dataset_test_ReliefF_ANOVA_filtered.csv')
        test_filtered.to_csv(test_path_filtered, index=False)

        print("Filtered datasets saved.")
    except Exception as e:
        print(f"Error saving datasets: {e}")

    # Save reports
    try:
        report_path = os.path.join(output_folder, 'relieff_anova_selection_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(selector.get_feature_report())

        config_path = os.path.join(output_folder, 'relieff_anova_configuration.txt')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("RELIEFF + ANOVA + CORRELATION - CONFIGURATION\n")
            f.write("="*50 + "\n\n")
            for key, value in CONFIG.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nRESULTS:\n")
            f.write(f"Original features: {X_train_raw.shape[1]}\n")
            f.write(f"Selected features: {len(selector.selected_features)}\n")
            f.write(f"Total reduction: {reduction_pct:.1f}%\n")

        print("Reports saved.")
    except Exception as e:
        print(f"Error saving reports: {e}")

    print("\nPROCESS COMPLETED WITH RELIEFF + ANOVA + CORRELATION!")
    print(f"Results saved in: {output_folder}/")
    print(f"Plots saved in: {output_folder}/plots/")


# Execute main function
if __name__ == "__main__":
    main()
