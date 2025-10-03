"""
TRIPLE UMAP GENERATOR - Uses Already Processed Datasets
====================================================================
Generates 3 types of UMAPs using datasets saved by the selection code:
1. Original Data (time series as features - each point = 1 feature)
2. Transformed Data (all statistical features)
3. Selected Data (features ranked by filters)

CORRECT: Original time series are converted into direct features!
"""

import os
import pandas as pd
import numpy as np
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import ast

warnings.filterwarnings('ignore')

# ====================================================================
# MAIN CONFIGURATION
# ====================================================================

CONFIG = {
    # Choose the dataset
    'dataset_type': 'train',  # OPTIONS: 'train' or 'test'
    
    # UMAP settings
    'umap_settings': {
        'n_neighbors': 25,      # Neighbors for graph construction
        'min_dist': 0.1,       # Minimum distance between points
        'n_components': 2,     # Output dimensions (always 2 for visualization)
        'random_state': 42     # Seed for reproducibility
    },
    
    # Plot settings
    'plot_settings': {
        'width': 800,
        'height': 600,
        'title_size': 22,      # AUMENTADO
        'font_size': 18,       # ADICIONADO para controle de fonte geral
        'point_size': 8,
        'opacity': 0.7
    },
    
    # Input and output folders
    'folders': {
        'processed_folder': './Preprocessed_Data',
        'filtered_folder': './ReliefF_ANOVA_Correlation_Results',
        'original_dataset': './dataset_robot.csv',
        'output_folder': './UMAP_Analysis',
    }
}

# ====================================================================
# AUXILIARY FUNCTIONS
# ====================================================================

def display_config():
    """Shows current configuration"""
    print(f"\n{'='*60}")
    print("TRIPLE UMAP GENERATOR CONFIGURATION")
    print(f"{'='*60}")
    print(f"Dataset chosen: {CONFIG['dataset_type'].upper()}")
    print(f"UMAP neighbors: {CONFIG['umap_settings']['n_neighbors']}")
    print(f"UMAP min_dist: {CONFIG['umap_settings']['min_dist']}")
    print(f"Processed folder: {CONFIG['folders']['processed_folder']}")
    print(f"Filtered folder: {CONFIG['folders']['filtered_folder']}")
    print(f"Output folder: {CONFIG['folders']['output_folder']}")
    print(f"Plot size: {CONFIG['plot_settings']['width']}x{CONFIG['plot_settings']['height']}")
    print(f"Font size: {CONFIG['plot_settings']['font_size']}")
    print(f"Uses datasets already processed by feature selection code")

def create_umap_plot(data, labels, title, filename):
    """
    Creates a UMAP plot and saves it
    """
    print(f"   Generating {title}...")
    
    # UMAP configuration
    reducer = umap.UMAP(**CONFIG['umap_settings'])
    
    # Apply UMAP
    embedding = reducer.fit_transform(data)
    
    # Create DataFrame for plotting
    umap_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    umap_df['Class'] = labels.reset_index(drop=True)
    
    # Create plot
    fig = px.scatter(
        umap_df,
        x='UMAP_1',
        y='UMAP_2',
        color='Class',
        title=title,
        width=CONFIG['plot_settings']['width'],
        height=CONFIG['plot_settings']['height'],
        opacity=CONFIG['plot_settings']['opacity'],
        hover_data={'Class': True}
    )
    
    # Customize the plot
    fig.update_traces(marker_size=CONFIG['plot_settings']['point_size'])
    fig.update_layout(
        title_font_size=CONFIG['plot_settings']['title_size'],
        title_x=0.5,
        font=dict(size=CONFIG['plot_settings']['font_size']), # AQUI A ALTERAÇÃO
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        )
    )
    
    # Save
    output_path = os.path.join(CONFIG['folders']['output_folder'], filename)
    fig.write_html(output_path)
    print(f"     Saved: {filename}")
    
    # Embedding statistics
    print(f"     Embedding shape: {embedding.shape}")
    print(f"     UMAP_1 range: [{embedding[:, 0].min():.2f}, {embedding[:, 0].max():.2f}]")
    print(f"     UMAP_2 range: [{embedding[:, 1].min():.2f}, {embedding[:, 1].max():.2f}]")
    
    return embedding, umap_df

def load_all_processed_datasets():
    """
    Loads the 3 datasets already processed by the feature selection code,
    with folder adjustment for the selected dataset.
    """
    print(f"\nLOADING THE 3 PROCESSED DATASETS ({CONFIG['dataset_type'].upper()})...")
    
    processed_folder = CONFIG['folders']['processed_folder']
    filtered_folder = CONFIG['folders']['filtered_folder']
    dataset_type = CONFIG['dataset_type']
    
    # Paths for the 3 already processed files
    if dataset_type == 'train':
        original_path = os.path.join(processed_folder, 'dataset_train_original.csv')
        transformed_path = os.path.join(processed_folder, 'dataset_train_transformed.csv')
        selected_path = os.path.join(filtered_folder, 'dataset_train_ReliefF_ANOVA_filtered.csv')
    else:  # test
        original_path = os.path.join(processed_folder, 'dataset_test_original.csv')
        transformed_path = os.path.join(processed_folder, 'dataset_test_transformed.csv')
        selected_path = os.path.join(filtered_folder, 'dataset_test_ReliefF_ANOVA_filtered.csv')
    
    try:
        # Load the 3 already processed datasets
        print(f"   Loading original dataset from: {os.path.basename(original_path)}")
        df_original = pd.read_csv(original_path)
        
        print(f"   Loading transformed dataset from: {os.path.basename(transformed_path)}")
        df_transformed = pd.read_csv(transformed_path)
        
        print(f"   Loading selected dataset from: {os.path.basename(selected_path)}")
        df_selected = pd.read_csv(selected_path)
        
        print(f"   All datasets loaded:")
        print(f"      Original ({dataset_type}): {df_original.shape}")
        print(f"      Transformed ({dataset_type}): {df_transformed.shape}")
        print(f"      Selected Transformed ({dataset_type}): {df_selected.shape}")
        
        # ================================================================
        # PROCESS ORIGINAL DATA: CONVERT TIME SERIES INTO FEATURES
        # ================================================================
        print("\n   PROCESSING ORIGINAL DATA (TIME SERIES)...")
        
        # Check if still contains time series as strings
        time_series_columns = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        
        # Check if any of the time series columns exist and are not already numerical
        has_time_series_str = False
        for col in time_series_columns:
            if col in df_original.columns and df_original[col].dtype == 'object':
                first_non_nan = df_original[col].dropna().iloc[0] if not df_original[col].dropna().empty else None
                if isinstance(first_non_nan, str):
                    has_time_series_str = True
                    break
        
        if has_time_series_str:
            print("   Converting time series into features (each point = 1 feature)...")
            
            # Convert time series into direct features
            features_list = []
            
            for idx, row in df_original.iterrows():
                row_features = {'label': row['label']}
                
                for var in time_series_columns:
                    if var in row:
                        try:
                            # Parse time series
                            if isinstance(row[var], str):
                                values = ast.literal_eval(row[var])  # Convert string to list
                            else:
                                values = row[var] # Assume it's already a list/array
                            
                            # Each point in the series becomes a feature
                            if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                                for i, val in enumerate(values):
                                    row_features[f'{var}_t{i}'] = float(val)
                            else:
                                # If no valid data or empty, fill with zeros
                                for i in range(15):  # Assuming 15 points per series
                                    row_features[f'{var}_t{i}'] = 0.0
                                    
                        except Exception as e:
                            print(f"      WARNING: Error processing {var} in row {idx}: {e}")
                            # Default values in case of error
                            for i in range(15):  # Assuming 15 points per series
                                row_features[f'{var}_t{i}'] = 0.0
                
                features_list.append(row_features)
            
            # Create new DataFrame with time series as features
            df_original_processed = pd.DataFrame(features_list)
            print(f"   Time series converted! Format: {df_original_processed.shape}")
            # Ensure 90 features if 6 vars * 15 points
            expected_features = len(time_series_columns) * 15
            print(f"   {len(time_series_columns)} variables x 15 points = {expected_features} temporal features + label")
            
        else:
            print("   Original dataset already in correct format or doesn't contain time series for conversion.")
            df_original_processed = df_original.copy()
        
        # Check consistency of sizes
        if len(df_original_processed) != len(df_transformed) or len(df_transformed) != len(df_selected):
            print(f"   WARNING: Different sizes between datasets!")
            print(f"      Original: {len(df_original_processed)}, Transformed: {len(df_transformed)}, Selected: {len(df_selected)}")
            # Try to align indices to avoid future errors
            common_indices = df_original_processed.index.intersection(df_transformed.index).intersection(df_selected.index)
            if len(common_indices) > 0:
                df_original_processed = df_original_processed.loc[common_indices].reset_index(drop=True)
                df_transformed = df_transformed.loc[common_indices].reset_index(drop=True)
                df_selected = df_selected.loc[common_indices].reset_index(drop=True)
                print(f"      Datasets aligned for {len(common_indices)} samples.")
            else:
                print(f"      ERROR: Could not align datasets. Check inputs.")
                return None, None, None

        # Ensure 'label' column exists in all and labels are consistent
        for df_temp in [df_original_processed, df_transformed, df_selected]:
            if 'label' not in df_temp.columns:
                print(f"   ERROR: 'label' column not found in one of the datasets. Aborting.")
                return None, None, None
        
        # Validate that labels are the same (or can be treated as such)
        if not df_transformed['label'].equals(df_selected['label']):
            print(f"   WARNING: Labels in 'transformed' and 'selected' are different. Using labels from 'transformed'.")
        labels = df_transformed['label'] # Use as reference
        
        return df_original_processed, df_transformed, df_selected, labels
        
    except Exception as e:
        print(f"   ERROR loading datasets: {e}")
        print(f"   Check if files exist in expected paths:")
        print(f"      - Original: {original_path}")
        print(f"      - Transformed: {transformed_path}")
        print(f"      - Selected Transformed: {selected_path}")
        return None, None, None, None

def generate_triple_umaps():
    """
    Main function that generates the 3 UMAPs
    CORRECTED: Uses only datasets already processed by selection code
    """
    display_config()
    
    # Create output folder
    os.makedirs(CONFIG['folders']['output_folder'], exist_ok=True)
    
    # Load the 3 already processed datasets
    df_original, df_transformed, df_selected, labels = load_all_processed_datasets()
    
    if df_original is None:
        print("ERROR: Failed to load processed datasets. Aborting.")
        return
    
    print(f"\nGENERATING THE 3 UMAPS WITH ALREADY PROCESSED DATASETS...")
    
    # ================================================================
    # UMAP 1: ORIGINAL DATA (time series as features)
    # ================================================================
    print(f"\n1. UMAP 1: ORIGINAL DATA (TIME SERIES)")
    
    # Prepare original data (time series as features)
    X_original = df_original.drop(columns=['label'])
    
    print(f"   Features in original data: {X_original.shape[1]}")
    print(f"   Each temporal point is a feature (6 variables x 15 points = 90 features)")
    print(f"   First columns: {list(X_original.columns[:min(10, X_original.shape[1])])}")
    
    # Check and clean data
    print("   Cleaning original data...")
    
    # Handle NaN values
    nan_count = X_original.isnull().sum().sum()
    if nan_count > 0:
        print(f"      WARNING: Found {nan_count} NaN values - replacing with 0")
        X_original_clean = X_original.fillna(0)
    else:
        X_original_clean = X_original.copy()
        print("   No NaN found")
    
    # Normalize
    scaler_original = StandardScaler()
    X_original_scaled = scaler_original.fit_transform(X_original_clean)
    
    # Generate UMAP
    embedding_1, df_umap_1 = create_umap_plot(
        data=X_original_scaled,
        labels=labels,
        title=f"UMAP 1: Original Time Series ({CONFIG['dataset_type'].title()}) - {X_original.shape[1]} Temporal Points",
        filename=f'umap_1_time_series_{CONFIG["dataset_type"]}.html'
    )
    
    # ================================================================
    # UMAP 2: TRANSFORMED DATA (all statistical features)
    # ================================================================
    print(f"\n2. UMAP 2: TRANSFORMED DATA")
    
    # Prepare transformed data
    X_transformed = df_transformed.drop(columns=['label'])
    X_transformed_clean = X_transformed.fillna(X_transformed.median(numeric_only=True))
    
    print(f"   Features in transformed data: {X_transformed.shape[1]}")
    
    # Normalize
    scaler_transformed = StandardScaler()
    X_transformed_scaled = scaler_transformed.fit_transform(X_transformed_clean)
    
    # Generate UMAP
    embedding_2, df_umap_2 = create_umap_plot(
        data=X_transformed_scaled,
        labels=labels,
        title=f"UMAP 2: Transformed Data ({CONFIG['dataset_type'].title()}) - {X_transformed.shape[1]} Complete Statistical Features",
        filename=f'umap_2_transformed_data_{CONFIG["dataset_type"]}.html'
    )
    
    # ================================================================
    # UMAP 3: SELECTED DATA (ranked features)
    # ================================================================
    print(f"\n3. UMAP 3: SELECTED TRANSFORMED DATA")
    
    # Prepare selected data
    X_selected = df_selected.drop(columns=['label'])
    X_selected_clean = X_selected.fillna(X_selected.median(numeric_only=True))
    
    print(f"   Features in selected data: {X_selected.shape[1]}")
    
    # Normalize
    scaler_selected = StandardScaler()
    X_selected_scaled = scaler_selected.fit_transform(X_selected_clean)
    
    # Generate UMAP
    embedding_3, df_umap_3 = create_umap_plot(
        data=X_selected_scaled,
        labels=labels,
        title=f"UMAP 3: Selected Transformed Features ({CONFIG['dataset_type'].title()}) - {X_selected.shape[1]} Ranked Features",
        filename=f'umap_3_selected_features_{CONFIG["dataset_type"]}.html'
    )
    
    # ================================================================
    # COMPARATIVE PLOT (3 UMAPs in same row)
    # ================================================================
    print(f"\nGENERATING COMPARATIVE PLOT...")
    
    # Create subplot with the 3 UMAPs in the same row
    fig_comparison = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f'UMAP 1: Time Series ({X_original.shape[1]} points)',
            f'UMAP 2: Transformed Data ({X_transformed.shape[1]} features)', 
            f'UMAP 3: Selected Transformed Features ({X_selected.shape[1]} features)'
        ],
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Add the 3 UMAPs
    umaps_data = [
        (embedding_1, df_umap_1['Class'], "Time Series", 1, 1),
        (embedding_2, df_umap_2['Class'], "Transformed Data", 1, 2), 
        (embedding_3, df_umap_3['Class'], "Selected Transformed Features", 1, 3)
    ]
    
    colors = px.colors.qualitative.Set1
    unique_labels = labels.unique()
    
    for embedding, current_labels, name_for_legend, row, col in umaps_data:
        for i, label_val in enumerate(unique_labels):
            mask = current_labels == label_val
            fig_comparison.add_trace(
                go.Scatter(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    mode='markers',
                    name=f'{label_val}',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=8,
                        opacity=0.7
                    ),
                    legendgroup=f'group{label_val}',
                    showlegend=(col == 1)  # Show legend only in first plot
                ),
                row=row, col=col
            )
    
    # Configure layout
    fig_comparison.update_layout(
        height=600,
        width=1800,
        title_text=f"Data Evolution: Time Series → Transformed → Selected ({CONFIG['dataset_type'].title()})",
        title_x=0.5,
        title_font_size=CONFIG['plot_settings']['title_size'],
        font=dict(size=CONFIG['plot_settings']['font_size']) # AQUI A ALTERAÇÃO
    )
    
    # AJUSTE ADICIONADO AQUI: Aumentar o tamanho da fonte dos subtítulos
    for annotation in fig_comparison['layout']['annotations']:
        annotation['font']['size'] = CONFIG['plot_settings']['title_size'] - 2 # Use um tamanho ligeiramente menor que o título principal
    
    # Save comparative plot
    comparison_path = os.path.join(CONFIG['folders']['output_folder'], f'umap_evolution_comparison_{CONFIG["dataset_type"]}.html')
    fig_comparison.write_html(comparison_path)
    print(f"   Comparative plot saved: umap_evolution_comparison_{CONFIG['dataset_type']}.html")
    
    # ================================================================
    # SEPARATE VARIANCE PLOT
    # ================================================================
    print(f"\nGENERATING SEPARATE VARIANCE ANALYSIS PLOT...")
    
    # Calculate explained variance for dimensionality comparison
    print("   Calculating explained variance...")
    datasets_for_variance = [
        (X_original_scaled, "Time Series", X_original.shape[1]),
        (X_transformed_scaled, "Transformed", X_transformed.shape[1]),
        (X_selected_scaled, "Selected", X_selected.shape[1])
    ]
    
    variance_data = []
    for data, name, n_features in datasets_for_variance:
        # Ensure n_components is not greater than the number of features or samples - 1
        pca_n_components = min(2, data.shape[1], data.shape[0] - 1) 
        
        if pca_n_components < 1:
            variance_data.append({
                'name': name,
                'variance_2_components': 0.0,
                'total_features': n_features
            })
            continue

        pca = PCA(n_components=pca_n_components)
        pca.fit(data)
        
        if len(pca.explained_variance_ratio_) > 1:
            cumulative_variance_2_comp = np.sum(pca.explained_variance_ratio_[:2])
        elif len(pca.explained_variance_ratio_) == 1:
            cumulative_variance_2_comp = pca.explained_variance_ratio_[0]
        else:
            cumulative_variance_2_comp = 0.0
            
        variance_data.append({
            'name': name,
            'variance_2_components': cumulative_variance_2_comp,
            'total_features': n_features
        })
    
    # Create separate variance plot
    fig_variance = go.Figure()
    
    fig_variance.add_trace(
        go.Bar(
            x=[d['name'] for d in variance_data],
            y=[d['variance_2_components'] for d in variance_data],
            name='Explained Variance (2 components)',
            marker_color='lightblue',
            text=[f"{d['total_features']} features<br>{d['variance_2_components']:.1%}" for d in variance_data],
            textposition='auto'
        )
    )
    
    fig_variance.update_layout(
        title=f"Explained Variance by First 2 PCA Components ({CONFIG['dataset_type'].title()})",
        xaxis_title="Dataset Type",
        yaxis_title="Explained Variance Ratio",
        height=500,
        width=800,
        title_x=0.5,
        title_font_size=CONFIG['plot_settings']['title_size'],
        font=dict(size=CONFIG['plot_settings']['font_size']) # AQUI A ALTERAÇÃO
    )
    
    # Save variance plot
    variance_path = os.path.join(CONFIG['folders']['output_folder'], f'variance_analysis_{CONFIG["dataset_type"]}.html')
    fig_variance.write_html(variance_path)
    print(f"   Variance analysis plot saved: variance_analysis_{CONFIG['dataset_type']}.html")
    
    # ================================================================
    # FINAL REPORT
    # ================================================================
    print(f"\n{'='*60}")
    print("TRIPLE UMAP GENERATION COMPLETED")
    print(f"{'='*60}")
    
    print(f"\nCONFIGURATION USED:")
    print(f"   Dataset: {CONFIG['dataset_type'].upper()}")
    print(f"   UMAP neighbors: {CONFIG['umap_settings']['n_neighbors']}")
    print(f"   UMAP min_dist: {CONFIG['umap_settings']['min_dist']}")
    
    print(f"\nDATA EVOLUTION (USING ALREADY PROCESSED DATASETS):")
    print(f"   1. Original Time Series → {X_original.shape[1]} temporal points (6 vars x 15 points)")
    print(f"   2. Transformed Data → {X_transformed.shape[1]} complete statistical features")
    print(f"   3. Selected Transformed Features → {X_selected.shape[1]} features ranked by filters")
    print(f"   Samples: {len(labels)} with {len(labels.unique())} classes ({list(labels.unique())})")
    
    # Show dimensionality reduction
    reduction_2 = ((X_transformed.shape[1] - X_selected.shape[1]) / X_transformed.shape[1]) * 100
    
    print(f"\nDIMENSIONALITY REDUCTION:")
    print(f"   Time Series: {X_original.shape[1]} points (6 variables x 15 points each)")
    print(f"   Complete Transformed: {X_transformed.shape[1]} statistical features")
    print(f"   Selected: {X_selected.shape[1]} features (reduction of {reduction_2:.1f}%)")
    
    print(f"\nGENERATED FILES:")
    print(f"   umap_1_time_series_{CONFIG['dataset_type']}.html")
    print(f"   umap_2_transformed_data_{CONFIG['dataset_type']}.html") 
    print(f"   umap_3_selected_features_{CONFIG['dataset_type']}.html")
    print(f"   umap_evolution_comparison_{CONFIG['dataset_type']}.html")
    print(f"   variance_analysis_{CONFIG['dataset_type']}.html")
    
    print(f"\nLocation: {os.path.abspath(CONFIG['folders']['output_folder'])}")
    
    print(f"\nPROCESS COMPLETED!")
    print(f"The UMAPs show complete evolution using already processed datasets:")
    print(f"   Time Series ({X_original.shape[1]}) → Transformed ({X_transformed.shape[1]}) → Selected Transformed ({X_selected.shape[1]})")
    print(f"   Original data: time series as direct features (each point = 1 feature)")
    print(f"   No statistics - uses raw time series converted to features!")
    
    return {
        'embeddings': {
            'original': embedding_1,
            'transformed': embedding_2, 
            'selected': embedding_3
        },
        'dataframes': {
            'original': df_umap_1,
            'transformed': df_umap_2,
            'selected': df_umap_3
        },
        'variance_analysis': variance_data,
        'feature_counts': {
            'original': X_original.shape[1],
            'transformed': X_transformed.shape[1],
            'selected': X_selected.shape[1]
        }
    }

# ====================================================================
# EXECUTION
# ====================================================================

if __name__ == "__main__":
    # TO CHANGE CONFIGURATIONS, MODIFY THE CONFIG SECTION AT THE TOP
    
    print("STARTING TRIPLE UMAP GENERATOR (USING PROCESSED DATASETS)...")
    print("USES DATA ALREADY PROCESSED BY FEATURE SELECTION CODE!")
    
    # Execute with current configuration
    results = generate_triple_umaps()
    
    print(f"\nTO GENERATE WITH TEST DATA, CHANGE:")
    print(f"   CONFIG['dataset_type'] = 'test'")
    print(f"   And run again!")
    
    print(f"\nOTHER AVAILABLE CONFIGURATIONS:")
    print(f"   • umap_settings: adjust UMAP parameters")
    print(f"   • plot_settings: customize plot appearance")
    print(f"   • folders: change file locations")
    
    print(f"\nREQUIRED FILES (generated by selection code):")
    print(f"   dataset_train_original.csv / dataset_test_original.csv")
    print(f"   dataset_train_transformed.csv / dataset_test_transformed.csv")
    print(f"   dataset_train_ReliefF_ANOVA_filtered.csv / dataset_test_ReliefF_ANOVA_filtered.csv")

# ====================================================================
# ALTERNATIVE USAGE EXAMPLE
# ====================================================================

def generate_for_both_datasets():
    """
    Helper function to generate UMAPs for TRAIN AND TEST at once
    """
    print("GENERATING UMAPS FOR TRAIN AND TEST...")
    
    # Save original configuration
    original_dataset_type = CONFIG['dataset_type']
    
    results = {}
    
    # Generate for train
    CONFIG['dataset_type'] = 'train'
    print(f"\n{'='*30} TRAIN {'='*30}")
    results['train'] = generate_triple_umaps()
    
    # Generate for test  
    CONFIG['dataset_type'] = 'test'
    print(f"\n{'='*30} TEST {'='*30}")
    results['test'] = generate_triple_umaps()
    
    # Restore configuration
    CONFIG['dataset_type'] = original_dataset_type
    
    print(f"\nALL UMAPS GENERATED!")
    print(f"   6 individual UMAPs + 2 comparative plots + 2 variance plots")
    print(f"   Total: 10 HTML files generated")
    print(f"   Complete evolution: Time Series → Transformed → Selected")
    print(f"   Original data: time series as direct features!")
    
    return results

# Uncomment the line below to generate for both datasets:
# results_both = generate_for_both_datasets()
