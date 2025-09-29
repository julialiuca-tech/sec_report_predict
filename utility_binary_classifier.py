#!/usr/bin/env python3
"""
Utility functions for binary classification modeling

This module contains generic binary classification functions that can be reused across
different machine learning projects and experiments. It provides tools for:
- Data splitting strategies
- Correlation analysis
- Threshold analysis and confidence metrics
- Model evaluation and visualization
- Performance plotting (ROC, PR curves, confusion matrices)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
 


def _random_split(df, train_prop, prefix="", random_seed=None):
    """
    Helper function to perform random splitting.
    
    Args:
        df (pd.DataFrame): Full dataset
        train_prop (float): Proportion of data to use for training
        prefix (str): Prefix for print statements (e.g., "Fallback - ")
        random_seed (int, optional): Random seed for reproducibility. If None, uses truly random results.
        
    Returns:
        tuple: (df_train, df_val) - Training and validation DataFrames
    """
    print(f"\n🔄 {prefix}Splitting data randomly ({train_prop*100:.1f}% training, {(1-train_prop)*100:.1f}% validation)...")
    
    # Get total number of samples
    total_samples = len(df)
    train_size = int(total_samples * train_prop)
    
    # Create random indices for training
    if random_seed is not None:
        np.random.seed(random_seed)  # For reproducibility
    train_indices = np.random.choice(total_samples, size=train_size, replace=False)
    
    # Create boolean masks
    train_mask = np.zeros(total_samples, dtype=bool)
    train_mask[train_indices] = True
    val_mask = ~train_mask
    
    print(f"📊 {prefix}Training: {train_mask.sum()} samples ({train_mask.sum()/total_samples*100:.1f}%)")
    print(f"📊 {prefix}Validation: {val_mask.sum()} samples ({val_mask.sum()/total_samples*100:.1f}%)")
    
    # Return DataFrames instead of masks
    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    return df_train, df_val


def split_train_val_by_column(df, train_prop, by_column, split_for_training='random', random_seed=None):
    """
    Split data into training and validation sets based on specified column.
    
    Args:
        df (pd.DataFrame): Full dataset with metadata columns
        train_prop (float): Proportion of data to use for training (0.0 to 1.0)
        by_column (str): Column to split data by (None for random split)
        split_for_training (str): Strategy ('random', 'top', or 'bottom')
        random_seed (int, optional): Random seed for reproducibility. If None, uses truly random results.
        
    Returns:
        tuple: (df_train, df_val) - Training and validation DataFrames
    """
    # Validate split_for_training parameter
    if split_for_training not in ['random', 'top', 'bottom']:
        raise ValueError(f"Invalid split_for_training: '{split_for_training}'. Must be 'random', 'top' or 'bottom'.")
    
    # Validate train_prop parameter
    if not (0.0 <= train_prop <= 1.0):
        raise ValueError(f"train_prop must be between 0.0 and 1.0, got {train_prop}")
    
    if by_column is None: 
        return _random_split(df, train_prop, random_seed=random_seed)
    
    # Validate that by_column exists in the dataframe
    try:
        if by_column not in df.columns:
            raise KeyError(f"Column '{by_column}' not found in dataframe. Available columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error accessing column '{by_column}': {str(e)}")
        raise
    
    # Split by the specified column using cumulative count cutoff
    print(f"\n🔄 Splitting data by column '{by_column}' ({train_prop*100:.1f}% training, {(1-train_prop)*100:.1f}% validation)...")
    
    try: 
        
        # Get value counts for the specified column
        value_counts = df[by_column].value_counts()
        print(f"📊 Found {len(value_counts)} unique values in column '{by_column}'")
        
        # Prepare values based on split strategy
        if split_for_training == 'random':
            if random_seed is not None:
                np.random.seed(random_seed)  # For reproducibility
            shuffled_values = value_counts.index.values.copy()
            np.random.shuffle(shuffled_values)
        elif split_for_training == 'top':
            shuffled_values = value_counts.sort_index(ascending=False).index.values
        elif split_for_training == 'bottom':
            shuffled_values = value_counts.sort_index(ascending=True).index.values
        
        # Calculate total samples and target cutoff
        total_samples = len(df)
        target_cutoff = int(total_samples * train_prop)
        
        # Create a Series with shuffled values and their counts, then calculate cumulative sum
        shuffled_counts = value_counts[shuffled_values]
        cumulative_counts = shuffled_counts.cumsum()
        
        # Find the cutoff point: include values where cumulative count <= target_cutoff
        train_mask_cumsum = cumulative_counts <= target_cutoff
        
        # Get the values that should be in training set
        train_values = shuffled_counts[train_mask_cumsum].index.tolist()
        
        # Create train/validation masks
        train_mask = df[by_column].isin(train_values)
        val_mask = ~train_mask

        print(f"📊 Training: {train_mask.sum()} samples from {len(train_values)} {by_column} values ({train_mask.sum()/total_samples*100:.1f}%)")
        print(f"📊 Validation: {val_mask.sum()} samples from {len(value_counts) - len(train_values)} {by_column} values ({val_mask.sum()/total_samples*100:.1f}%)")
        
        # Return DataFrames instead of masks
        df_train = df[train_mask].copy()
        df_val = df[val_mask].copy()
        return df_train, df_val
        
    except Exception as e:
        print(f"❌ Error during column-based splitting: {str(e)}")
        print("🔄 Falling back to random splitting...")
        return _random_split(df, train_prop, prefix="Fallback - ", random_seed=random_seed)


def baseline_binary_classifier(X_train, X_val, y_train, y_val, model_name): 
    """
    Build and evaluate baseline binary classifiers.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        y_train (array-like): Training target labels
        y_val (array-like): Validation target labels
        model_name (str): Model type ('rf' or 'xgb')
        
    Returns:
        dict: Results with metrics and trained model
    """
    feature_cols = X_train.columns

    if model_name == 'rf':
        # Random Forest can handle missing values natively by using NaN as a separate category
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5, 
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    elif model_name == 'xgb':
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
    else:
        raise ValueError(f"Invalid model type: {model_name}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model_name': model_name,
        'trained_model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc, 
        'feature_importance': feature_importance
    }



def correlation_analysis(df, feature_cols, target_label):
    """
    Calculate correlations between features and target variable.
    
    Args:
        df (pd.DataFrame): Dataset with features and target labels  
        feature_cols (list): List of feature column names to analyze
        target_label (str): Name of the target column   
        
    Returns:
        pd.Series: Correlations sorted by absolute value (descending)
    """ 
    feature_cols = list(feature_cols)
    if len(feature_cols) == 0:
        print("❌ No feature columns found for correlation analysis.")
        return pd.DataFrame()
    
    print(f"📊 Analyzing correlations for {len(feature_cols)} features...")
    
    # Fill missing values with median for correlation calculation
    df_filled = df[feature_cols + [target_label]].fillna(df[feature_cols].median())
    
    # Calculate correlations with target
    correlations = df_filled[feature_cols].corrwith(df_filled[target_label]).abs().sort_values(ascending=False)
    
    # Display top 20 features
    print(f"\n🔍 Top 20 Features by Absolute Correlation with {target_label}:")
    print("=" * 80)
    for i, (feature, corr) in enumerate(correlations.head(20).items(), 1):
        direction = "📈" if df_filled[feature].corr(df_filled[target_label]) > 0 else "📉"
        print(f"{i:2d}. {feature:<40} {direction} {corr:.4f}")
    
    # Summary statistics
    print(f"\n📊 Correlation Summary:")
    print(f"  Mean absolute correlation: {correlations.mean():.4f}")
    print(f"  Max absolute correlation: {correlations.max():.4f}")
    print(f"  Features with |corr| > 0.1: {(correlations > 0.1).sum()}")
    print(f"  Features with |corr| > 0.05: {(correlations > 0.05).sum()}")
    
    return correlations


def calc_FPR_FNR_at_confidence_threshold(y_val, y_pred_proba, threshold):
    """
    Calculate FPR and FNR at a specific threshold.
    
    Args:
        y_val (array-like): True binary labels (0 or 1)
        y_pred_proba (array-like): Predicted probabilities for positive class
        threshold (float): Decision threshold (0.0 to 1.0)
        
    Returns:
        dict: Metrics including FPR, FNR, TPR, TNR, precision, accuracy
    """
    # Create binary predictions
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_val, y_pred_binary)
    TN, FP, FN, TP = cm.ravel()
    
    # Calculate rates
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Sensitivity)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate (Specificity)
    
    # Additional metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return {
        'threshold': threshold,
        'FPR': FPR,
        'FNR': FNR,
        'TPR': TPR,
        'TNR': TNR,
        'precision': precision,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }


def analyze_confidence_thresholds(y_val, y_pred_proba, thresholds=[0.25, 0.5, 0.75, 0.9]):
    """
    Analyze model performance at different confidence thresholds.
    
    Args:
        y_val (array-like): True binary labels (0 or 1)
        y_pred_proba (array-like): Predicted probabilities for positive class
        thresholds (list): List of decision thresholds to evaluate
        
    Returns:
        list: List of dictionaries with metrics for each threshold
    """
    
    results = []
    for threshold in thresholds:
        result = calc_FPR_FNR_at_confidence_threshold(y_val, y_pred_proba, threshold)
        results.append(result)
        
        # Calculate number of predicted positives
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        num_predicted_positives = y_pred_binary.sum()
        total_samples = len(y_pred_binary)
        
        print(f"\nThreshold: {threshold}")
        print(f"  Predicted Positives: {num_predicted_positives}/{total_samples} ({num_predicted_positives/total_samples*100:.1f}%)")
        print(f"  FPR: {result['FPR']:.4f} ({result['FPR']*100:.2f}%)")
        print(f"  FNR: {result['FNR']:.4f} ({result['FNR']*100:.2f}%)")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
    
    return results

def plot_roc_curve(y_val, y_pred_proba):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_val (array-like): True binary labels (0 or 1)
        y_pred_proba (array-like): Predicted probabilities for positive class
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

    # Calculate AUC (Area Under Curve)
    roc_auc = auc(fpr, tpr)


    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_precision_recall_curve_with_percentile_marker(y_val, y_pred_proba, percentile=10):
    """
    Plot precision-recall curve with marker at specified percentile threshold.
    
    Args:
        y_val (array-like): True binary labels (0 or 1)
        y_pred_proba (array-like): Predicted probabilities for positive class
        percentile (int): Percentile threshold to mark (default 10)
        
    Returns:
        dict: Threshold, precision, and recall at the specified percentile
    """
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    ap_score = average_precision_score(y_val, y_pred_proba)
    
    # Calculate percentile threshold
    threshold_percentile = np.percentile(y_pred_proba, percentile)
    
    # Calculate precision and recall at this threshold
    y_pred_binary = (y_pred_proba >= threshold_percentile).astype(int)
    precision_at_threshold = precision_score(y_val, y_pred_binary)
    recall_at_threshold = recall_score(y_val, y_pred_binary)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AP = {ap_score:.3f})')
    
    # Add marker at percentile threshold
    plt.plot(recall_at_threshold, precision_at_threshold, 'ro', markersize=12, 
             label=f'{percentile}th percentile (P={precision_at_threshold:.3f}, R={recall_at_threshold:.3f})')
    
    # Add threshold value as text
    plt.annotate(f'Threshold: {threshold_percentile:.3f}', 
                xy=(recall_at_threshold, precision_at_threshold),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'threshold': threshold_percentile,
        'precision': precision_at_threshold,
        'recall': recall_at_threshold
    }


def plot_confusion_matrix(y_val, y_pred_proba):
    """
    Plot confusion matrix for binary classification.
    
    Args:
        y_val (array-like): True binary labels (0 or 1)
        y_pred_proba (array-like): Predicted probabilities for positive class
    """
    import seaborn as sns
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_val, y_pred)

    # Plot confusion matrix
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_feature_importance(model, X_train, y_train):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained tree-based model with feature_importances_ attribute
        X_train (pd.DataFrame): Training features DataFrame
        y_train (array-like): Training target labels (not used but kept for consistency)
    """
    # Calculate feature importance
    feature_importance = model.feature_importances_

    # Plot feature importance
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(feature_importance)), feature_importance, align='center')
    plt.yticks(range(len(feature_importance)), X_train.columns)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    plt.show()

def plot_proba_hist_by_class(y_val, y_pred_proba):
    """
    Plot probability histogram separated by true class.
    
    Args:
        y_val (array-like): True binary labels (0 or 1)
        y_pred_proba (array-like): Predicted probabilities for positive class
    """
    # Plot probability histogram by class
    plt.figure(figsize=(8, 6))
    plt.hist(y_pred_proba[y_val == 1], bins=20, alpha=0.5, label='Class 1')
    plt.hist(y_pred_proba[y_val == 0], bins=20, alpha=0.5, label='Class 0')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Histogram by Class')
    plt.legend(loc="upper right")
    plt.show()

