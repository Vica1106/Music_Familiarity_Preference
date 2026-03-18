#!/usr/bin/env python3
"""
EEG Classifier Pipeline for Music Familiarity and Enjoyment Prediction

This script performs exploratory machine learning classification on EEG band-power features.

Usage:
    python run_eeg_classifiers.py --task familiarity
    python run_eeg_classifiers.py --task enjoyment
    python run_eeg_classifiers.py --task familiarity --log-transform
    python run_eeg_classifiers.py --feature-set combined
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# File paths
ALPHA_REGION_CSV = "Code/analysis_jessie/alpha_region.csv"
THETA_CSV = "familarity_thetawave/results/theta_by_familiarity.csv"
OUTPUT_DIR = "results"

# Standardize column names
COLUMN_MAPPING = {
    'subject': 'Subject',
    'song': 'Song_ID',
    'song_id': 'Song_ID',
    'trial': 'Song_ID',
    'familiarity': 'Familiarity',
    'enjoyment': 'Enjoyment',
    'theta_frontal': 'theta_frontal',
    'theta_central': 'theta_central',
    'theta_parietal': 'theta_parietal',
    'frontal_theta': 'theta_frontal',
    'alpha_frontal': 'alpha_frontal',
    'alpha_central': 'alpha_central',
    'alpha_parietal': 'alpha_parietal',
    'frontal': 'alpha_frontal',
    'central': 'alpha_central',
    'parietal': 'alpha_parietal',
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_alpha_data():
    """
    Load alpha band-power data from CSV files.
    
    Returns:
        DataFrame with Subject, Song_ID, Enjoyment, and regional alpha features
    """
    print("Loading alpha data...")
    
    # Load alpha region data (contains z-scored regional alpha)
    alpha_df = pd.read_csv(ALPHA_REGION_CSV, index_col=0)
    
    # Standardize column names
    alpha_df.columns = [c.strip() for c in alpha_df.columns]
    
    # Rename columns to standard names
    rename_dict = {}
    for col in alpha_df.columns:
        col_lower = col.lower()
        if col_lower == 'subject':
            rename_dict[col] = 'Subject'
        elif col_lower == 'song_id':
            rename_dict[col] = 'Song_ID'
        elif col_lower == 'enjoyment':
            rename_dict[col] = 'Enjoyment'
        elif col_lower == 'frontal':
            rename_dict[col] = 'alpha_frontal'
        elif col_lower == 'central':
            rename_dict[col] = 'alpha_central'
        elif col_lower == 'parietal':
            rename_dict[col] = 'alpha_parietal'
    
    alpha_df = alpha_df.rename(columns=rename_dict)
    
    # Select relevant columns
    cols_to_keep = ['Subject', 'Song_ID', 'Enjoyment', 'alpha_frontal', 'alpha_central', 'alpha_parietal']
    available_cols = [c for c in cols_to_keep if c in alpha_df.columns]
    alpha_df = alpha_df[available_cols]
    
    print(f"  Loaded alpha data: {len(alpha_df)} rows, columns: {list(alpha_df.columns)}")
    return alpha_df


def load_theta_data():
    """
    Load theta band-power data from CSV file.
    
    Returns:
        DataFrame with Subject, Song_ID, Enjoyment, Familiarity, and regional theta features
    """
    print("Loading theta data...")
    
    theta_df = pd.read_csv(THETA_CSV)
    
    # Standardize column names
    theta_df.columns = [c.strip() for c in theta_df.columns]
    
    # Rename columns to standard names
    rename_dict = {}
    for col in theta_df.columns:
        col_lower = col.lower()
        if col_lower == 'subject':
            rename_dict[col] = 'Subject'
        elif col_lower == 'song_id':
            rename_dict[col] = 'Song_ID'
        elif col_lower == 'enjoyment':
            rename_dict[col] = 'Enjoyment'
        elif col_lower == 'familiarity':
            rename_dict[col] = 'Familiarity'
        elif col_lower in ['theta_frontal', 'theta_central', 'theta_parietal']:
            rename_dict[col] = col_lower
    
    theta_df = theta_df.rename(columns=rename_dict)
    
    # Select relevant columns
    cols_to_keep = ['Subject', 'Song_ID', 'Enjoyment', 'Familiarity', 'theta_frontal', 'theta_central', 'theta_parietal']
    available_cols = [c for c in cols_to_keep if c in theta_df.columns]
    theta_df = theta_df[available_cols]
    
    print(f"  Loaded theta data: {len(theta_df)} rows, columns: {list(theta_df.columns)}")
    return theta_df


def build_feature_table(alpha_df, theta_df):
    """
    Merge alpha and theta data into a unified feature table.
    
    Args:
        alpha_df: Alpha band-power DataFrame
        theta_df: Theta band-power DataFrame
    
    Returns:
        Merged DataFrame with all features
    """
    print("\nBuilding unified feature table...")
    
    # Merge on Subject and Song_ID
    merged = pd.merge(
        theta_df, 
        alpha_df, 
        on=['Subject', 'Song_ID'], 
        how='inner',
        suffixes=('_theta', '_alpha')
    )
    
    # Handle Enjoyment column (may come from either source)
    if 'Enjoyment_theta' in merged.columns and 'Enjoyment_alpha' in merged.columns:
        # Use theta Enjoyment as primary (has more data)
        merged['Enjoyment'] = merged['Enjoyment_theta']
        merged = merged.drop(columns=['Enjoyment_theta', 'Enjoyment_alpha'])
    elif 'Enjoyment_theta' in merged.columns:
        merged = merged.rename(columns={'Enjoyment_theta': 'Enjoyment'})
    elif 'Enjoyment_alpha' in merged.columns:
        merged = merged.rename(columns={'Enjoyment_alpha': 'Enjoyment'})
    
    print(f"  Merged feature table: {len(merged)} rows")
    print(f"  Columns: {list(merged.columns)}")
    
    # Save merged feature table
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "ml_feature_table.csv")
    merged.to_csv(output_path, index=False)
    print(f"  Saved feature table to: {output_path}")
    
    return merged


# ============================================================================
# Label Preparation Functions
# ============================================================================

def prepare_familiarity_labels(df):
    """
    Prepare binary familiarity labels.
    
    Low familiarity = ratings 1-2
    High familiarity = rating 5
    Drop ratings 3 and 4
    
    Args:
        df: Feature DataFrame
    
    Returns:
        DataFrame with binary labels added
    """
    df = df.copy()
    
    # Filter to keep only Low (1-2) and High (5) familiarity
    df = df[df['Familiarity'].isin([1, 2, 5])].copy()
    
    # Create binary label: 0 = Low (1-2), 1 = High (5)
    df['label'] = (df['Familiarity'] == 5).astype(int)
    
    print(f"  Familiarity labels: {len(df)} samples")
    print(f"    Low (1-2): {(df['label'] == 0).sum()}")
    print(f"    High (5): {(df['label'] == 1).sum()}")
    
    return df


def prepare_enjoyment_labels(df):
    """
    Prepare binary enjoyment labels.
    
    Low enjoyment = ratings 1-2
    High enjoyment = ratings 4-5
    Drop rating 3
    
    If only extreme values (1 vs 5) exist, use those.
    
    Args:
        df: Feature DataFrame
    
    Returns:
        DataFrame with binary labels added
    """
    df = df.copy()
    
    # Check what enjoyment values exist
    unique_vals = sorted(df['Enjoyment'].unique())
    print(f"  Enjoyment values in data: {unique_vals}")
    
    if 3 in unique_vals and 4 in unique_vals:
        # Use 1-2 vs 4-5
        df = df[df['Enjoyment'].isin([1, 2, 4, 5])].copy()
        df['label'] = (df['Enjoyment'] >= 4).astype(int)
    elif 5 in unique_vals:
        # Fall back to 1 vs 5 if available
        df = df[df['Enjoyment'].isin([1, 5])].copy()
        df['label'] = (df['Enjoyment'] == 5).astype(int)
    elif 4 in unique_vals:
        # Use 1-2 vs 4
        df = df[df['Enjoyment'].isin([1, 2, 4])].copy()
        df['label'] = (df['Enjoyment'] >= 4).astype(int)
    else:
        # Use whatever is available
        df['label'] = (df['Enjoyment'] >= 3).astype(int)
    
    print(f"  Enjoyment labels: {len(df)} samples")
    print(f"    Low: {(df['label'] == 0).sum()}")
    print(f"    High: {(df['label'] == 1).sum()}")
    
    return df


# ============================================================================
# Feature Selection
# ============================================================================

def get_feature_sets(df):
    """
    Get different feature set options.
    
    Args:
        df: Feature DataFrame
    
    Returns:
        Dictionary of feature set names to column lists
    """
    feature_sets = {}
    
    # Theta only features
    theta_cols = [c for c in ['theta_frontal', 'theta_central', 'theta_parietal'] if c in df.columns]
    if theta_cols:
        feature_sets['theta'] = theta_cols
    
    # Alpha only features
    alpha_cols = [c for c in ['alpha_frontal', 'alpha_central', 'alpha_parietal'] if c in df.columns]
    if alpha_cols:
        feature_sets['alpha'] = alpha_cols
    
    # Combined features
    if theta_cols and alpha_cols:
        feature_sets['combined'] = theta_cols + alpha_cols
    elif theta_cols:
        feature_sets['combined'] = theta_cols
    elif alpha_cols:
        feature_sets['combined'] = alpha_cols
    
    # Frontal only (if available)
    frontal_cols = [c for c in ['theta_frontal', 'alpha_frontal'] if c in df.columns]
    if frontal_cols:
        feature_sets['frontal'] = frontal_cols
    
    return feature_sets


def select_features(df, feature_set_name, feature_sets):
    """
    Select features based on feature set name.
    
    Args:
        df: Feature DataFrame
        feature_set_name: Name of feature set to use
        feature_sets: Dictionary of available feature sets
    
    Returns:
        X (feature matrix), feature_names
    """
    if feature_set_name not in feature_sets:
        print(f"Warning: Feature set '{feature_set_name}' not found. Using 'combined'.")
        feature_set_name = 'combined'
    
    feature_cols = feature_sets[feature_set_name]
    X = df[feature_cols].values
    feature_names = feature_cols
    
    return X, feature_names


# ============================================================================
# Model Training and Evaluation
# ============================================================================

def get_models():
    """
    Get baseline classifier models.
    
    Returns:
        Dictionary of model names to model instances
    """
    models = {
        'LogReg': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            solver='lbfgs'
        ),
        'SVM': SVC(
            kernel='rbf', 
            random_state=42,
            C=1.0,
            gamma='scale'
        ),
        'RF': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            n_jobs=-1
        )
    }
    return models


def evaluate_model(X, y, model, model_name, cv=5, use_log_transform=False):
    """
    Evaluate a model using stratified k-fold cross-validation.
    
    Args:
        X: Feature matrix
        y: Labels
        model: Sklearn model instance
        model_name: Name of the model
        cv: Number of folds
        use_log_transform: Whether to log-transform features
    
    Returns:
        Dictionary of metrics
    """
    X_processed = X.copy()
    
    # Optional log-transform (add small constant to avoid log(0))
    # Only apply to positive power features (theta), not z-scored features (alpha)
    if use_log_transform:
        # Check if features are all positive (power features)
        all_positive = np.all(X_processed > 0)
        if all_positive:
            X_processed = np.log10(X_processed + 1e-12)
        else:
            print(f"    Warning: Some features have non-positive values, applying log only to positive columns.")
            # Apply log only to positive columns (those with all values > 0)
            for col in range(X_processed.shape[1]):
                if np.all(X_processed[:, col] > 0):
                    X_processed[:, col] = np.log10(X_processed[:, col] + 1e-12)
    
    # Create pipeline with scaling inside CV
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # Stratified k-fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Get predictions using cross_val_predict
    y_pred = cross_val_predict(pipeline, X_processed, y, cv=skf)
    
    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='binary')
    cm = confusion_matrix(y, y_pred)
    
    # Calculate per-fold metrics
    fold_accs = []
    fold_f1s = []
    for train_idx, test_idx in skf.split(X_processed, y):
        X_train, X_test = X_processed[train_idx], X_processed[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit and predict
        pipeline.fit(X_train, y_train)
        y_fold_pred = pipeline.predict(X_test)
        
        fold_accs.append(accuracy_score(y_test, y_fold_pred))
        fold_f1s.append(f1_score(y_test, y_fold_pred, average='binary'))
    
    metrics = {
        'model': model_name,
        'accuracy_mean': np.mean(fold_accs),
        'accuracy_std': np.std(fold_accs),
        'f1_mean': np.mean(fold_f1s),
        'f1_std': np.std(fold_f1s),
        'confusion_matrix': cm,
        'y_pred': y_pred
    }
    
    return metrics


def run_classification(df, task, feature_set_name='combined', use_log_transform=False):
    """
    Run complete classification pipeline.
    
    Args:
        df: Feature DataFrame
        task: 'familiarity' or 'enjoyment'
        feature_set_name: Name of feature set to use
        use_log_transform: Whether to log-transform power features
    
    Returns:
        List of metrics dictionaries
    """
    print(f"\n{'='*60}")
    print(f"Classification Task: {task.upper()}")
    print(f"Feature Set: {feature_set_name}")
    print(f"Log Transform: {use_log_transform}")
    print(f"{'='*60}")
    
    # Prepare labels
    if task == 'familiarity':
        df_labeled = prepare_familiarity_labels(df)
    else:
        df_labeled = prepare_enjoyment_labels(df)
    
    if len(df_labeled) < 10:
        print(f"Error: Not enough samples for classification: {len(df_labeled)}")
        return []
    
    # Get feature sets
    feature_sets = get_feature_sets(df_labeled)
    print(f"\nAvailable feature sets: {list(feature_sets.keys())}")
    
    # Select features
    X, feature_names = select_features(df_labeled, feature_set_name, feature_sets)
    y = df_labeled['label'].values
    
    print(f"\nUsing {len(feature_names)} features: {feature_names}")
    print(f"Total samples: {len(X)}")
    
    # Get models
    models = get_models()
    
    # Evaluate each model
    results = []
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(X, y, model, model_name, cv=5, use_log_transform=use_log_transform)
        results.append(metrics)
        
        print(f"  Accuracy: {metrics['accuracy_mean']:.3f} (+/- {metrics['accuracy_std']:.3f})")
        print(f"  F1 Score: {metrics['f1_mean']:.3f} (+/- {metrics['f1_std']:.3f})")
        print(f"  Confusion Matrix:")
        print(f"    {metrics['confusion_matrix'].tolist()}")
    
    return results, feature_names, y


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_model_comparison(results, task, output_dir=OUTPUT_DIR):
    """
    Plot model accuracy comparison.
    
    Args:
        results: List of metrics dictionaries
        task: Task name (familiarity or enjoyment)
        output_dir: Output directory
    """
    models = [r['model'] for r in results]
    accuracies = [r['accuracy_mean'] for r in results]
    acc_stds = [r['accuracy_std'] for r in results]
    f1s = [r['f1_mean'] for r in results]
    f1_stds = [r['f1_std'] for r in results]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    ax1 = axes[0]
    x = np.arange(len(models))
    bars1 = ax1.bar(x, accuracies, yerr=acc_stds, capsize=5, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Chance (0.5)')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Mean CV Accuracy', fontsize=12)
    ax1.set_title(f'Model Accuracy Comparison - {task.capitalize()}', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # F1 plot
    ax2 = axes[1]
    bars2 = ax2.bar(x, f1s, yerr=f1_stds, capsize=5, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Chance (0.5)')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Mean CV F1 Score', fontsize=12)
    ax2.set_title(f'Model F1 Comparison - {task.capitalize()}', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, f1 in zip(bars2, f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'model_comparison_{task}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved model comparison plot to: {output_path}")
    plt.close()


def plot_confusion_matrices(results, y_true, task, output_dir=OUTPUT_DIR):
    """
    Plot confusion matrices for all models.
    
    Args:
        results: List of metrics dictionaries
        y_true: True labels
        task: Task name
        output_dir: Output directory
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, metrics in enumerate(results):
        model_name = metrics['model']
        cm = metrics['confusion_matrix']
        
        ax = axes[idx]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f'{model_name} - {task.capitalize()}')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'confusion_matrices_{task}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrices to: {output_path}")
    plt.close()


def plot_feature_importance(results, feature_names, task, output_dir=OUTPUT_DIR):
    """
    Plot feature importance from Random Forest.
    
    Args:
        results: List of metrics dictionaries
        feature_names: List of feature names
        task: Task name
        output_dir: Output directory
    """
    # Find Random Forest results
    rf_results = None
    for r in results:
        if r['model'] == 'RF':
            rf_results = r
            break
    
    if rf_results is None:
        print("No Random Forest results for feature importance plot.")
        return
    
    # Train RF to get feature importance
    # Note: This is a simplification - ideally we'd use the actual RF from CV
    print("\nNote: Feature importance requires retraining RF on full data for visualization.")
    print("This is a simplified visualization showing relative feature values.")
    
    # Create a simple bar plot showing mean feature values
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get feature means as proxy (actual importance requires retraining)
    # We'll skip this for now as it would require retraining
    print("Feature importance plot skipped - requires retraining RF on full data.")


def save_metrics(results, task, output_dir=OUTPUT_DIR):
    """
    Save metrics to CSV files.
    
    Args:
        results: List of metrics dictionaries
        task: Task name
        output_dir: Output directory
    """
    # Save main metrics
    metrics_df = pd.DataFrame([{
        'model': r['model'],
        'accuracy_mean': r['accuracy_mean'],
        'accuracy_std': r['accuracy_std'],
        'f1_mean': r['f1_mean'],
        'f1_std': r['f1_std']
    } for r in results])
    
    output_path = os.path.join(output_dir, f'ml_metrics_{task}.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"Saved metrics to: {output_path}")
    
    # Save confusion matrices
    for r in results:
        cm_df = pd.DataFrame(
            r['confusion_matrix'],
            index=['actual_low', 'actual_high'],
            columns=['pred_low', 'pred_high']
        )
        cm_path = os.path.join(output_dir, f'confusion_matrix_{task}_{r["model"]}.csv')
        cm_df.to_csv(cm_path)
        print(f"Saved confusion matrix to: {cm_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point for the classifier pipeline."""
    parser = argparse.ArgumentParser(
        description='EEG Classifier for Music Familiarity and Enjoyment Prediction'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='familiarity',
        choices=['familiarity', 'enjoyment'],
        help='Classification task: familiarity or enjoyment'
    )
    parser.add_argument(
        '--feature-set',
        type=str,
        default='combined',
        choices=['theta', 'alpha', 'combined', 'frontal'],
        help='Feature set to use'
    )
    parser.add_argument(
        '--log-transform',
        action='store_true',
        help='Apply log-transform to power features'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("EEG Classifier Pipeline")
    print("="*60)
    
    # Load data
    alpha_df = load_alpha_data()
    theta_df = load_theta_data()
    
    # Build unified feature table
    feature_df = build_feature_table(alpha_df, theta_df)
    
    # Run classification
    results, feature_names, y_true = run_classification(
        feature_df, 
        task=args.task,
        feature_set_name=args.feature_set,
        use_log_transform=args.log_transform
    )
    
    if not results:
        print("\nClassification failed. Exiting.")
        return
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    chance = 0.5
    for r in results:
        model = r['model']
        acc = r['accuracy_mean']
        f1 = r['f1_mean']
        print(f"{model}: Accuracy = {acc:.3f}, F1 = {f1:.3f}")
    
    # Check if any model beats chance
    best_acc = max(r['accuracy_mean'] for r in results)
    if best_acc > chance:
        print(f"\n[OK] Best model beats chance level ({chance:.1f})")
    else:
        print(f"\n[--] No model beats chance level ({chance:.1f})")
    
    # Save outputs
    print("\nSaving outputs...")
    save_metrics(results, args.task)
    plot_model_comparison(results, args.task)
    plot_confusion_matrices(results, y_true, args.task)
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == '__main__':
    main()
