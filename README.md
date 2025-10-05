# Feature-selection-with-interpretability-failure-classification-in-robotic-manipulators
Multi-class classification of robotic failure types using force and torque time series data with advanced feature selection techniques.
# Robot Collision Classification using Force/Torque Sensors

## Overview

This project implements a comprehensive machine learning pipeline for classifying collision types in robotic systems using F/T (Force/Torque) sensor data. The pipeline includes:

- Statistical feature extraction from time series
- Advanced feature selection (ReliefF + ANOVA + Correlation)
- Multiple ML models (KNN, SVM, MLP, LightGBM)
- SHAP-based interpretability analysis
- Comprehensive validation framework

## Key Features

- **3 Classification Scenarios**:
  - Scenario 1: Raw sensor data (90 temporal features)
  - Scenario 2: Full statistical features (144 features)
  - Scenario 3: Selected features via ReliefF+ANOVA+Correlation (54 features)

- **Feature Selection Pipeline**:
  - ReliefF algorithm for non-linear relationships
  - ANOVA F-test for statistical significance
  - Correlation-based redundancy removal
  - Cumulative contribution analysis

- **Model Interpretability**:
  - SHAP (SHapley Additive exPlanations) analysis
  - Feature importance ranking per model
  - Visualization of decision boundaries

## Dataset

- **Source**: 5 independent robotic experiments (LP1-LP5)
- **Total Samples**: 252
- **Classes**: 3 collision types
- **Features**: 6 time series (Fx, Fy, Fz, Tx, Ty, Tz) with ~15 points each
- **Split**: 80% train (201 samples) / 20% test (51 samples)

### Class Distribution
- **Class 1**: collision in part (134 samples - 53.2%)
- **Class 2**: slightly moved (87 samples - 34.5%)
- **Class 3**: collision in tool (31 samples - 12.3%)

## Installation
```bash

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
