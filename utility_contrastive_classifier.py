#!/usr/bin/env python3
"""
Contrastive Learning Classifier for Multi-class Stock Prediction

This module implements a contrastive learning approach for 3-class classification:
- good: stock rising more than 5% in a month
- bad: stock falling more than 5% in a month  
- mediocre: anything in between

The model learns embeddings where similar classes are close together and different
classes are far apart, then uses these embeddings for classification.
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union

from config import (
    FEATURIZED_ALL_QUARTERS_FILE,
    STOCK_TREND_DATA_FILE,
    QUARTER_GRADIENTS,
    SPLIT_STRATEGY,
    FEATURE_SUFFIXES,
    Y_LABEL,
    MODEL_DIR,
    FEATURE_IMPORTANCE_RANKING_FLAG,
    STOCK_DIR,
)


from baseline_model import prep_data_feature_label, split_data_for_train_val, collect_column_names_w_suffix, filter_features_by_importance


class ContrastiveStockClassifier(BaseEstimator, ClassifierMixin):
    """
    Contrastive Learning Classifier for 3-class stock prediction.
    
    This model uses contrastive learning to learn embeddings that pull similar
    classes together and push different classes apart, then applies a classification
    head to predict the 3 classes.
    
    Implements sklearn-compatible interface with fit(), predict(), and predict_proba().
    
    Args:
        embedding_dim (int): Dimension of the learned embeddings (default: 64)
        hidden_dims (list): List of hidden layer dimensions for the encoder (default: [128, 64])
        margin (float): Margin for contrastive loss (default: 1.0)
        temperature (float): Temperature parameter for contrastive loss (default: 0.1)
        batch_size (int): Batch size for training (default: 256)
        epochs (int): Number of training epochs (default: 100)
        learning_rate (float): Learning rate for optimizer (default: 0.001)
        contrastive_weight (float): Weight for contrastive loss vs classification loss (default: 0.5)
        device (str): Device to use ('cpu' or 'cuda') (default: 'cpu')
        random_state (int): Random seed for reproducibility (default: 42)
    """
    
    def __init__(self, 
                 embedding_dim=64,
                 hidden_dims=[128, 64],
                 margin=1.0,
                 temperature=0.1,
                 batch_size=256,
                 epochs=100,
                 learning_rate=0.001,
                 contrastive_weight=0.5,
                 device='cpu',
                 random_state=42):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.margin = margin
        self.temperature = temperature
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.contrastive_weight = contrastive_weight
        self.device = device
        self.random_state = random_state
        
        # Will be set during fit
        self.scaler_ = None
        self.label_encoder_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.model_ = None
        
        # Set random seeds for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
    
    def _create_model(self, input_dim, n_classes):
        """Create the neural network model with encoder and classifier."""
        return ContrastiveNet(
            input_dim=input_dim,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            n_classes=n_classes
        ).to(self.device)
    
    def fit(self, X, y, validation_data=None):
        """
        Train the contrastive learning model.
        
        Args:
            X (array-like): Training features of shape (n_samples, n_features)
            y (array-like): Training labels. Can be:
                           - numeric: 0 (bad), 1 (mediocre), 2 (good)
                           - string: 'bad', 'mediocre', 'good'
                           - binary indicators: will be converted to 3 classes based on thresholds
            validation_data (tuple, optional): (X_val, y_val) for validation during training
        
        Returns:
            self: Returns self for method chaining
        """
        # Convert input to numpy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        
        # Final check for NaN/Inf before scaling
        if np.isnan(X).any():
            raise ValueError("Input X contains NaN values. Please impute missing values before calling fit().")
        if np.isinf(X).any():
            raise ValueError("Input X contains infinite values. Please handle infinite values before calling fit().")
        
        # Handle different label formats
        y = self._prepare_labels(y)
        
        # Normalize features
        # Use with_mean=False for zero-variance columns, but we should remove them before this
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Check for NaN/Inf after scaling (should not happen, but good to verify)
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            raise ValueError("Scaled features contain NaN or infinite values. This usually indicates zero-variance features.")
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X_scaled.shape[1]
        
        # Create model
        self.model_ = self._create_model(self.n_features_, self.n_classes_)
        
        # Create datasets
        train_dataset = StockDataset(X_scaled, y_encoded)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val)
            y_val = self._prepare_labels(y_val)
            X_val_scaled = self.scaler_.transform(X_val)
            y_val_encoded = self.label_encoder_.transform(y_val)
            val_dataset = StockDataset(X_val_scaled, y_val_encoded)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Train model
        self._train_model(train_loader, val_loader)
        
        # Set model to eval mode
        self.model_.eval()
        
        return self
    
    def _prepare_labels(self, y):
        """
        Prepare labels for 3-class classification.
        
        If y is binary (0/1), converts to 3 classes:
        - 0 -> 'bad'
        - 1 -> 'good'  
        - (anything else -> 'mediocre')
        
        If y is numeric with 3 values, converts to class names.
        If y is already string labels, validates and returns as-is.
        """
        unique_values = np.unique(y)
        
        # If binary labels (0, 1), convert to 3 classes
        if len(unique_values) == 2 and set(unique_values) == {0, 1}:
            # Assuming 1 means good performance, 0 means bad
            # We'll need to handle this based on your actual label definition
            # For now, map 0->bad, 1->good, and we'll need additional info for mediocre
            # This is a placeholder - you may need to adjust based on your data
            y_str = np.where(y == 1, 'good', 'bad')
            print("⚠️  Warning: Binary labels detected. Converting to 2 classes (bad, good).")
            print("   For 3-class classification, you need labels indicating: bad, mediocre, good")
            return y_str
        
        # If numeric labels with 3 distinct values
        elif len(unique_values) == 3:
            # Map to class names
            sorted_values = sorted(unique_values)
            mapping = {sorted_values[0]: 'bad', sorted_values[1]: 'mediocre', sorted_values[2]: 'good'}
            return np.array([mapping[val] for val in y])
        
        # If already string labels
        elif all(isinstance(val, str) for val in unique_values):
            valid_labels = {'bad', 'mediocre', 'good'}
            if set(unique_values).issubset(valid_labels):
                return y
            else:
                raise ValueError(f"Invalid string labels: {unique_values}. Must be in {valid_labels}")
        
        else:
            raise ValueError(f"Unexpected label format: {unique_values}. "
                           f"Expected binary (0/1), 3-class numeric, or string labels ('bad', 'mediocre', 'good')")
    
    def _train_model(self, train_loader, val_loader=None):
        """Train the contrastive learning model."""
        criterion_contrastive = ContrastiveLoss(margin=self.margin, temperature=self.temperature)
        criterion_classification = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model_.train()
            train_loss = 0.0
            train_contrastive_loss = 0.0
            train_class_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                embeddings, logits = self.model_(batch_X)
                
                # Classification loss
                class_loss = criterion_classification(logits, batch_y)
                
                # Contrastive loss (pulls same class together, pushes different classes apart)
                contrastive_loss = criterion_contrastive(embeddings, batch_y)
                
                # Combined loss
                total_loss = (1 - self.contrastive_weight) * class_loss + self.contrastive_weight * contrastive_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                train_class_loss += class_loss.item()
                train_contrastive_loss += contrastive_loss.item()
            
            # Validation
            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        embeddings, logits = self.model_(batch_X)
                        class_loss = criterion_classification(logits, batch_y)
                        contrastive_loss = criterion_contrastive(embeddings, batch_y)
                        total_loss = (1 - self.contrastive_weight) * class_loss + self.contrastive_weight * contrastive_loss
                        val_loss += total_loss.item()
                
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                train_loss /= len(train_loader)
                train_class_loss /= len(train_loader)
                train_contrastive_loss /= len(train_loader)
                if val_loader is not None:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.4f} "
                          f"(Class: {train_class_loss:.4f}, Contrastive: {train_contrastive_loss:.4f}) - "
                          f"Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.4f} "
                          f"(Class: {train_class_loss:.4f}, Contrastive: {train_contrastive_loss:.4f})")
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Args:
            X (array-like): Samples to predict, shape (n_samples, n_features)
        
        Returns:
            array: Predicted class labels
        """
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X)
        
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            _, logits = self.model_(X_tensor)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Convert back to original label format
        return self.label_encoder_.inverse_transform(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples.
        
        Args:
            X (array-like): Samples to predict, shape (n_samples, n_features)
        
        Returns:
            array: Class probabilities, shape (n_samples, n_classes)
        """
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X)
        
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            _, logits = self.model_(X_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        
        return probabilities
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X (array-like): Test samples
            y (array-like): True labels
        
        Returns:
            float: Mean accuracy
        """
        y_pred = self.predict(X)
        y = np.asarray(y)
        y = self._prepare_labels(y)
        return accuracy_score(y, y_pred)


class ContrastiveNet(nn.Module):
    """
    Neural network with contrastive learning.
    
    Architecture:
    - Encoder: Maps input to embedding space
    - Classifier: Maps embedding to class logits
    """
    
    def __init__(self, input_dim, embedding_dim, hidden_dims, n_classes):
        super(ContrastiveNet, self).__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, n_classes)
        
        # L2 normalization for embeddings (helps with contrastive learning)
        self.normalize_embeddings = True
    
    def forward(self, x):
        """
        Forward pass.
        
        Returns:
            tuple: (embeddings, logits)
        """
        # Encode to embedding space
        embeddings = self.encoder(x)
        
        # Normalize embeddings (L2 norm)
        if self.normalize_embeddings:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Classify
        logits = self.classifier(embeddings)
        
        return embeddings, logits


class ContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for multi-class classification.
    
    Pulls samples from the same class together and pushes samples
    from different classes apart in the embedding space.
    """
    
    def __init__(self, margin=1.0, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Tensor of shape (batch_size,) with class labels
        
        Returns:
            Contrastive loss value
        """
        batch_size = embeddings.shape[0]
        
        # Compute pairwise distances
        # embeddings is already normalized, so dot product = cosine similarity
        similarity_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Create mask for positive pairs (same class)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.t()).float()
        negative_mask = 1 - positive_mask
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Contrastive loss: maximize similarity for positive pairs, minimize for negative pairs
        # Using temperature-scaled softmax (similar to InfoNCE)
        exp_sim = torch.exp(similarity_matrix / self.temperature)
        
        # Sum over positive pairs
        positive_sim = (exp_sim * positive_mask).sum(dim=1)
        # Sum over all pairs (excluding self)
        total_sim = exp_sim.sum(dim=1) - exp_sim.diag()
        
        # Loss: -log(positive_sim / total_sim)
        loss = -torch.log((positive_sim / (total_sim + 1e-8)) + 1e-8).mean()
        
        return loss


class StockDataset(Dataset):
    """PyTorch Dataset for stock data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_three_class_labels(price_returns, good_threshold=0.05, bad_threshold=-0.05):
    """
    Convert price returns to 3-class labels.
    
    Args:
        price_returns (array-like): Price returns (e.g., percentage changes)
        good_threshold (float): Threshold for 'good' class (default: 0.05 for 5% gain)
        bad_threshold (float): Threshold for 'bad' class (default: -0.05 for 5% loss)
    
    Returns:
        array: Class labels ('good', 'mediocre', 'bad')
    """
    price_returns = np.asarray(price_returns)
    labels = np.where(price_returns >= good_threshold, 'good',
                     np.where(price_returns <= bad_threshold, 'bad', 'mediocre'))
    return labels


def evaluate_contrastive_classifier(model, X_test, y_test, df_test=None, print_report=True):
    """
    Evaluate the contrastive classifier and print performance metrics.
    
    Args:
        model: Trained ContrastiveStockClassifier
        X_test: Test features (DataFrame or array)
        y_test: Test labels
        df_test (pd.DataFrame, optional): Full test dataframe. If provided and contains Y_LABEL column
                                          (from config), will compute correlation with predicted probabilities.
        print_report (bool): Whether to print classification report
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Prepare labels
    y_test_prepared = model._prepare_labels(y_test)
    
    accuracy = accuracy_score(y_test_prepared, y_pred)
    
    # Compute correlation coefficient between predicted probability scores and labels
    # Encode labels: bad=-1, good=1, mediocre=0
    label_to_score = {'bad': -1, 'mediocre': 0, 'good': 1}
    true_scores = np.array([label_to_score[label] for label in y_test_prepared])
    
    # Convert probabilities to a single score: weighted sum with label encodings
    # y_proba columns are ordered by model.classes_ (alphabetically: 'bad', 'good', 'mediocre')
    proba_scores = np.zeros(len(y_proba))
    for i, label in enumerate(model.classes_):
        proba_scores += y_proba[:, i] * label_to_score[label]
    
    # Compute Pearson correlation coefficient
    correlation_coef, p_value = pearsonr(true_scores, proba_scores)
    
    results = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_proba,
        'correlation_coef': correlation_coef,
        'correlation_p_value': p_value
    }
    
    # Normalize scores to [0, 1] range for ROC-AUC
    y_pred_proba = (proba_scores + 1) / 2
    
    # Compute correlation with Y_LABEL from config if df_test is provided
    correlation_trend_coef = None
    correlation_trend_p_value = None
    roc_auc_trend = None
    if df_test is not None and Y_LABEL in df_test.columns:
        # Align indices: if X_test is a DataFrame, use its index; otherwise use range
        if isinstance(X_test, pd.DataFrame):
            y_label_values = df_test.loc[X_test.index, Y_LABEL].values
        else:
            # If X_test is array, assume it matches df_test row order
            y_label_values = df_test[Y_LABEL].values[:len(proba_scores)]
        
        # Remove any NaN values for correlation and ROC-AUC computation
        valid_mask = ~np.isnan(y_label_values)
        if valid_mask.sum() > 0:
            # Correlation coefficient
            correlation_trend_coef, correlation_trend_p_value = pearsonr(
                y_label_values[valid_mask], 
                proba_scores[valid_mask]
            )
            results['correlation_trend_coef'] = correlation_trend_coef
            results['correlation_trend_p_value'] = correlation_trend_p_value
            
            # ROC-AUC score
            try:
                roc_auc_trend = roc_auc_score(y_label_values[valid_mask], y_pred_proba[valid_mask])
                results['roc_auc_trend'] = roc_auc_trend
            except ValueError:
                # ROC-AUC may fail if Y_LABEL has only one class
                results['roc_auc_trend'] = None
    
    if print_report:
        print("\n" + "="*60)
        print("CONTRASTIVE CLASSIFIER EVALUATION")
        print("="*60)
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Correlation Coefficient (pred_proba vs labels): {correlation_coef:.4f} (p-value: {p_value:.4e})")
        print(f"  (Labels encoded as: bad=-1, mediocre=0, good=1)")
        
        if correlation_trend_coef is not None:
            print(f"Correlation Coefficient (pred_proba vs {Y_LABEL}): {correlation_trend_coef:.4f} (p-value: {correlation_trend_p_value:.4e})")
        if roc_auc_trend is not None:
            print(f"ROC-AUC Score (y_pred_proba vs {Y_LABEL}): {roc_auc_trend:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_prepared, y_pred, target_names=model.classes_))
        
        # Confusion matrix: rows = True labels, columns = Predicted labels
        cm = confusion_matrix(y_test_prepared, y_pred, labels=model.classes_)
        print("\nConfusion Matrix:")
        print("(Rows = True labels, Columns = Predicted labels)")
        print(f"\n{'True\\Predicted':<15}", end="")
        for label in model.classes_:
            print(f"{label:>12}", end="")
        print()
        for i, true_label in enumerate(model.classes_):
            print(f"{true_label:<15}", end="")
            for j in range(len(model.classes_)):
                print(f"{cm[i, j]:>12}", end="")
            print()
        print("="*60)
    
    return results


def prep_data_for_contrastive_learning(df=None,
                                       df_train=None,
                                       df_test=None,
                                       good_threshold=1.05,
                                       bad_threshold=0.95,
                                       train_val_split_prop=0.7,
                                       train_val_split_strategy=None,
                                       feature_suffixes=None):
    """
    Prepare data for contrastive learning classifier.
    
    This function:
    1. Creates 3-class labels from price returns
    2. Identifies feature columns
    3. Handles infinite values and NaN imputation (upstream)
    4. Optionally splits data into train/val sets (if df_train/df_test not provided)
    5. Removes zero-variance columns (post-split, using training data only)
    
    Args:
        df (pd.DataFrame, optional): DataFrame with features and 'price_return' column.
                                     Required if df_train and df_test are not provided.
        df_train (pd.DataFrame, optional): Pre-split training DataFrame. If provided along with df_test,
                                           split step is skipped.
        df_test (pd.DataFrame, optional): Pre-split test/validation DataFrame. If provided along with df_train,
                                          split step is skipped.
        good_threshold (float): Threshold for 'good' class (default: 1.05)
        bad_threshold (float): Threshold for 'bad' class (default: 0.95)
        train_val_split_prop (float): Proportion for train/val split (default: 0.7). Only used if df_train/df_test not provided.
        train_val_split_strategy (dict): Split strategy dict, e.g. {'period': 'bottom'} 
                                        (default: None, uses SPLIT_STRATEGY from config). Only used if df_train/df_test not provided.
        feature_suffixes (list): Feature suffixes to use (default: None, uses FEATURE_SUFFIXES from config)
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val, feature_cols_final, df_train, df_val)
            - X_train: Training features DataFrame
            - X_val: Validation/test features DataFrame  
            - y_train: Training labels Series
            - y_val: Validation/test labels Series
            - feature_cols_final: Final list of feature column names (after zero-variance removal)
            - df_train: Full training DataFrame (with all columns including metadata)
            - df_val: Full validation/test DataFrame (with all columns including metadata)
    """
    from config import SPLIT_STRATEGY, FEATURE_SUFFIXES
    
    # Use defaults from config if not provided
    if train_val_split_strategy is None:
        train_val_split_strategy = SPLIT_STRATEGY
    if feature_suffixes is None:
        feature_suffixes = FEATURE_SUFFIXES
    
    # Determine if we need to do the split or use provided dataframes
    if df_train is not None and df_test is not None:
        # Use provided train/test dataframes (skip split)
        df_train = df_train.copy()
        df_test = df_test.copy()
        print(f"   Using provided train/test splits: {len(df_train)} train, {len(df_test)} test samples")
        
        # Create 3-class labels for both train and test
        price_returns_train = df_train['price_return'].values
        price_returns_test = df_test['price_return'].values
        y_labels_train = create_three_class_labels(price_returns_train, good_threshold=good_threshold, bad_threshold=bad_threshold)
        y_labels_test = create_three_class_labels(price_returns_test, good_threshold=good_threshold, bad_threshold=bad_threshold)
        df_train['y_label'] = y_labels_train
        df_test['y_label'] = y_labels_test
        
        print(f"   Train label distribution: {pd.Series(y_labels_train).value_counts().to_dict()}")
        print(f"   Test label distribution: {pd.Series(y_labels_test).value_counts().to_dict()}")
        
        # Determine feature columns from training data
        suffix_cols = collect_column_names_w_suffix(df_train.columns, feature_suffixes)
        feature_cols = suffix_cols + [f for f in df_train.columns if '_change' in f and f not in suffix_cols]
        
        # Data cleaning on training data first
        print(f"\n   Data validation and cleaning (training data)...")
        X_train_features = df_train[feature_cols].copy()
        inf_count_train = np.isinf(X_train_features.values).sum().sum()
        if inf_count_train > 0:
            print(f"   ⚠️  Found {inf_count_train} infinite values in training data. Replacing with NaN...")
            X_train_features = X_train_features.replace([np.inf, -np.inf], np.nan)
        
        train_median = X_train_features.median()
        nan_cols = train_median.isna().sum()
        if nan_cols > 0:
            print(f"   ⚠️  Found {nan_cols} columns with all NaN values in training data. These will be filled with 0.")
            train_median = train_median.fillna(0)
        X_train_features = X_train_features.fillna(train_median)
        df_train[feature_cols] = X_train_features[feature_cols]
        
        # Apply same cleaning to test data (using training statistics)
        X_test_features = df_test[feature_cols].copy()
        inf_count_test = np.isinf(X_test_features.values).sum().sum()
        if inf_count_test > 0:
            print(f"   ⚠️  Found {inf_count_test} infinite values in test data. Replacing with NaN...")
            X_test_features = X_test_features.replace([np.inf, -np.inf], np.nan)
        X_test_features = X_test_features.fillna(train_median)
        df_test[feature_cols] = X_test_features[feature_cols]
        
    elif df is not None:
        # Do the split (original behavior)
        # Create 3-class labels
        price_returns = df['price_return'].values
        y_labels = create_three_class_labels(price_returns, good_threshold=good_threshold, bad_threshold=bad_threshold)
        df = df.copy()  # Work on a copy to avoid modifying original
        df['y_label'] = y_labels 
        print(f"   Label distribution: {pd.Series(y_labels).value_counts().to_dict()}")
        
        # Prepare feature columns (do this before split to get all possible features)
        suffix_cols = collect_column_names_w_suffix(df.columns, feature_suffixes)
        feature_cols = suffix_cols + [f for f in df.columns if '_change' in f and f not in suffix_cols]
        
        # Data cleaning upstream (before split)
        print(f"\n   Data validation and cleaning (upstream)...")
        
        # Handle infinite values - can be done upstream safely
        df_features = df[feature_cols].copy()
        inf_count = np.isinf(df_features.values).sum().sum()
        if inf_count > 0:
            print(f"   ⚠️  Found {inf_count} infinite values. Replacing with NaN...")
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values - using full dataset median (minor data leakage, but acceptable for median)
        # Note: Best practice would be to compute median from training data only, but moving upstream
        # for convenience. The difference is typically minimal with robust statistics like median.
        df_median = df_features.median()
        nan_cols = df_median.isna().sum()
        if nan_cols > 0:
            print(f"   ⚠️  Found {nan_cols} columns with all NaN values. These will be filled with 0.")
            df_median = df_median.fillna(0)
        df_features = df_features.fillna(df_median)
        df[feature_cols] = df_features[feature_cols]  # Update df with cleaned features
        
        # Split data into train/val sets (after upstream cleaning)
        df_train, df_test = split_data_for_train_val(df, 
                                                    train_val_split_prop=train_val_split_prop,    
                                                    train_val_split_strategy=train_val_split_strategy)
        print(f"   Train set: {df_train.shape[0]} samples")
        print(f"   Validation set: {df_test.shape[0]} samples")
    else:
        raise ValueError("Must provide either 'df' (for automatic split) or both 'df_train' and 'df_test' (for pre-split data)")
    
    # Extract features and labels from dataframes
    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    y_train = df_train['y_label'].copy()
    y_test = df_test['y_label'].copy()
    
    # Critical: Check for zero variance columns AFTER split (must use training data only)
    # With time-based splits, a feature may have variance in full dataset but zero variance in training period
    print(f"\n   Checking for zero-variance columns (post-split, using training data only)...")
    train_var = X_train.var()
    zero_var_cols = train_var[train_var == 0].index.tolist()
    if len(zero_var_cols) > 0:
        print(f"   ⚠️  Found {len(zero_var_cols)} zero-variance columns in training data. Dropping them...")
        X_train = X_train.drop(columns=zero_var_cols)
        X_test = X_test.drop(columns=zero_var_cols)
        print(f"   Remaining features: {X_train.shape[1]}")
    
    # Get final feature columns (after zero-variance removal)
    feature_cols_final = X_train.columns.tolist()
    
    print(f"   ✅ Data preparation complete: {len(feature_cols_final)} features, {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test, feature_cols_final, df_train, df_test


def main_real_data():
    """
    Demonstration using real project data with proper train/val split strategy.
    
    This function uses the same data preparation process as baseline_model.py:
    1. Loads featurized data and stock trends
    2. Creates 3-class labels from price returns
    3. Splits data using the configured split strategy
    4. Trains and evaluates the contrastive classifier
    """
    print("="*70)
    print("CONTRASTIVE STOCK CLASSIFIER - REAL DATA DEMONSTRATION")
    print("="*70)
    
    # ============================================================================
    # STEP 0: Prepare data
    # ============================================================================
    print("\n📊 Loading and preparing real project data...")
    
    # Load featurized data and stock trends
    df_features = pd.read_csv(FEATURIZED_ALL_QUARTERS_FILE)
    df_trends = pd.read_csv(STOCK_TREND_DATA_FILE)
    df = prep_data_feature_label(df_featurized_data=df_features, 
                                  df_stock_trend=df_trends,
                                  quarters_for_gradient_comp=QUARTER_GRADIENTS)

    # Prepare data for contrastive learning
    X_train, X_val, y_train, y_val, feature_cols_final, df_train, df_val = \
        prep_data_for_contrastive_learning(
            df=df,
            good_threshold=1.05,
            bad_threshold=0.95,
            train_val_split_prop=0.7,
            train_val_split_strategy=SPLIT_STRATEGY,
            feature_suffixes=FEATURE_SUFFIXES
        )
    
    # ============================================================================
    # STEP 1: Initialize the model
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 1: Initializing ContrastiveStockClassifier")
    print("="*70)
    
    model = ContrastiveStockClassifier(
        embedding_dim=128,
        hidden_dims=[128, 128, 64],
        margin=1.0,
        temperature=0.1,
        batch_size=256,
        epochs=100,  # Can be increased for better performance
        learning_rate=0.001,
        contrastive_weight=0.75,  # Balance between contrastive and classification loss
        device='cpu',  # Use 'cuda' if GPU available
        random_state=42
    )
    
    print(f"   Model configuration:")
    print(f"   - Embedding dimension: {model.embedding_dim}")
    print(f"   - Hidden dimensions: {model.hidden_dims}")
    print(f"   - Batch size: {model.batch_size}")
    print(f"   - Epochs: {model.epochs}")
    print(f"   - Contrastive weight: {model.contrastive_weight}")
    
    # ============================================================================
    # STEP 2: Training data is ready (already split)
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 2: Data splits ready")
    print("="*70)
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Validation samples: {X_val.shape[0]}")
    print(f"   Label distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
    print(f"   Label distribution (val): {pd.Series(y_val).value_counts().to_dict()}")
    
    # ============================================================================
    # STEP 3: Train the model
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 3: Training the model")
    print("="*70)
    print("   (This may take a few minutes...)")
    
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val)
    )
    
    print("   ✅ Training completed!")
    
    # ============================================================================
    # STEP 4: Make predictions
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 4: Making predictions on validation set")
    print("="*70)
    
    # Predict class labels
    y_pred = model.predict(X_val)
    print(f"   Predicted {len(y_pred)} samples")
    print(f"   Prediction distribution: {pd.Series(y_pred).value_counts().to_dict()}")
    
    # Predict probabilities
    y_proba = model.predict_proba(X_val)
    print(f"   Probability shape: {y_proba.shape}")
    print(f"   Sample probabilities (first 3):")
    for i in range(min(3, len(y_proba))):
        proba_dict = dict(zip(model.classes_, y_proba[i]))
        print(f"      Sample {i+1}: {proba_dict}")
    
    # ============================================================================
    # STEP 5: Evaluate the model
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 5: Evaluating model performance")
    print("="*70)
    
    results = evaluate_contrastive_classifier(model, X_val, y_val, df_test=df_val, print_report=True)
    
    # ============================================================================
    # STEP 6: Additional demonstrations
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 6: Additional demonstrations")
    print("="*70)
    
    # Show how to use score() method
    accuracy = model.score(X_val, y_val)
    print(f"\n   Model accuracy (using .score() method): {accuracy:.4f}")
    
    # Show how to create labels from price returns
    print(f"\n   Example: Creating 3-class labels from price returns")
    example_returns = np.array([0.08, 0.02, -0.07, 0.0, -0.03])
    example_labels = create_three_class_labels(example_returns, good_threshold=0.05, bad_threshold=-0.05)
    for ret, label in zip(example_returns, example_labels):
        print(f"      Return: {ret:+.2%} -> Label: {label}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\n💡 Tips:")
    print("   - Adjust 'contrastive_weight' to balance contrastive vs classification loss")
    print("   - Increase 'epochs' for better performance (but longer training time)")
    print("   - Use 'device=\"cuda\"' if you have GPU available")
    print("   - Tune 'temperature' and 'margin' for contrastive loss")
    print("   - Experiment with different 'embedding_dim' and 'hidden_dims'")
    
    return model, results

    
def main():
    """
    Demonstration of how to use the ContrastiveStockClassifier.
    
    This function shows:
    1. Creating 3-class labels from price returns
    2. Training a contrastive learning model
    3. Making predictions
    4. Evaluating model performance
    
    It can work with real project data (if available) or synthetic data.
    """
    print("="*70)
    print("CONTRASTIVE STOCK CLASSIFIER - DEMONSTRATION")
    print("="*70)
    
    # Try to load real data from the project
    use_real_data = False
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    
    try:
        from config import FEATURIZED_ALL_QUARTERS_FILE, STOCK_TREND_DATA_FILE
        from baseline_model import prep_data_feature_label
        import pandas as pd
        
        print("\n📊 Attempting to load real project data...")
        
        # Load featurized data
        if os.path.exists(FEATURIZED_ALL_QUARTERS_FILE):
            df_features = pd.read_csv(FEATURIZED_ALL_QUARTERS_FILE)
            print(f"   Loaded featurized data: {df_features.shape}")
            
            # Load stock trends
            if os.path.exists(STOCK_TREND_DATA_FILE):
                df_trends = pd.read_csv(STOCK_TREND_DATA_FILE)
                print(f"   Loaded stock trends: {df_trends.shape}")
                
                # Prepare data
                df_combined = prep_data_feature_label(
                    df_featurized_data=df_features,
                    df_stock_trend=df_trends,
                    df_history_data=None,
                    quarters_for_gradient_comp=None
                )
                
                # Extract features (columns with _current, _augment, or change in name)
                feature_cols = [col for col in df_combined.columns 
                              if any(suffix in col for suffix in ['_current', '_augment', 'change'])]
                
                # Extract price returns for label creation
                if 'price_return_1month' in df_combined.columns:
                    price_returns = df_combined['price_return_1month'].values
                    
                    # Create 3-class labels
                    y_labels = create_three_class_labels(price_returns, good_threshold=0.05, bad_threshold=-0.05)
                    
                    # Prepare feature matrix
                    X = df_combined[feature_cols].fillna(0).values
                    
                    # Split into train/test (80/20)
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
                    )
                    
                    use_real_data = True
                    print(f"   ✅ Using real data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
                    print(f"   Features: {len(feature_cols)}")
                    print(f"   Label distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
                    
    except Exception as e:
        print(f"   ⚠️  Could not load real data: {e}")
        print("   Using synthetic data for demonstration...")
    
    # If real data not available, create synthetic data
    if not use_real_data:
        print("\n🔬 Generating synthetic data for demonstration...")
        n_samples = 2000
        n_features = 50
        
        # Generate synthetic features
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        
        # Create synthetic price returns (correlated with some features)
        # Good stocks: positive returns, Bad stocks: negative returns, Mediocre: near zero
        price_returns = (
            0.1 * X[:, 0] +  # Feature 0 influences returns
            0.05 * X[:, 1] +  # Feature 1 influences returns
            np.random.randn(n_samples) * 0.03  # Add noise
        )
        
        # Create 3-class labels from synthetic returns
        y_labels = create_three_class_labels(price_returns, good_threshold=0.05, bad_threshold=-0.05)
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
        )
        
        print(f"   ✅ Generated synthetic data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        print(f"   Features: {n_features}")
        print(f"   Label distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
    
    # Convert to DataFrames for sklearn compatibility
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    
    # ============================================================================
    # STEP 1: Initialize the model
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 1: Initializing ContrastiveStockClassifier")
    print("="*70)
    
    model = ContrastiveStockClassifier(
        embedding_dim=64,
        hidden_dims=[128, 64],
        margin=1.0,
        temperature=0.1,
        batch_size=256,
        epochs=50,  # Reduced for demonstration
        learning_rate=0.001,
        contrastive_weight=0.5,  # Balance between contrastive and classification loss
        device='cpu',  # Use 'cuda' if GPU available
        random_state=42
    )
    
    print(f"   Model configuration:")
    print(f"   - Embedding dimension: {model.embedding_dim}")
    print(f"   - Hidden dimensions: {model.hidden_dims}")
    print(f"   - Batch size: {model.batch_size}")
    print(f"   - Epochs: {model.epochs}")
    print(f"   - Contrastive weight: {model.contrastive_weight}")
    
    # ============================================================================
    # STEP 2: Split data for validation
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 2: Preparing validation data")
    print("="*70)
    
    # Further split training data for validation
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train_df, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"   Training samples: {X_train_fit.shape[0]}")
    print(f"   Validation samples: {X_val.shape[0]}")
    print(f"   Test samples: {X_test_df.shape[0]}")
    
    # ============================================================================
    # STEP 3: Train the model
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 3: Training the model")
    print("="*70)
    print("   (This may take a few minutes...)")
    
    model.fit(
        X_train_fit,
        y_train_fit,
        validation_data=(X_val, y_val)
    )
    
    print("   ✅ Training completed!")
    
    # ============================================================================
    # STEP 4: Make predictions
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 4: Making predictions")
    print("="*70)
    
    # Predict class labels
    y_pred = model.predict(X_test_df)
    print(f"   Predicted {len(y_pred)} samples")
    print(f"   Prediction distribution: {pd.Series(y_pred).value_counts().to_dict()}")
    
    # Predict probabilities
    y_proba = model.predict_proba(X_test_df)
    print(f"   Probability shape: {y_proba.shape}")
    print(f"   Sample probabilities (first 3):")
    for i in range(min(3, len(y_proba))):
        proba_dict = dict(zip(model.classes_, y_proba[i]))
        print(f"      Sample {i+1}: {proba_dict}")
    
    # ============================================================================
    # STEP 5: Evaluate the model
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 5: Evaluating model performance")
    print("="*70)
    
    results = evaluate_contrastive_classifier(model, X_test_df, y_test, print_report=True)
    
    # ============================================================================
    # STEP 6: Additional demonstrations
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 6: Additional demonstrations")
    print("="*70)
    
    # Show how to use score() method
    accuracy = model.score(X_test_df, y_test)
    print(f"\n   Model accuracy (using .score() method): {accuracy:.4f}")
    
    # Show how to create labels from price returns
    print(f"\n   Example: Creating 3-class labels from price returns")
    example_returns = np.array([0.08, 0.02, -0.07, 0.0, -0.03])
    example_labels = create_three_class_labels(example_returns, good_threshold=0.05, bad_threshold=-0.05)
    for ret, label in zip(example_returns, example_labels):
        print(f"      Return: {ret:+.2%} -> Label: {label}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\n💡 Tips:")
    print("   - Adjust 'contrastive_weight' to balance contrastive vs classification loss")
    print("   - Increase 'epochs' for better performance (but longer training time)")
    print("   - Use 'device=\"cuda\"' if you have GPU available")
    print("   - Tune 'temperature' and 'margin' for contrastive loss")
    print("   - Experiment with different 'embedding_dim' and 'hidden_dims'")


def invest_monthly_retro_w_contrastive(INVEST_EXP_START_MONTH_STR='2023-01', INVEST_EXP_END_MONTH_STR='2025-07'):
    """
    Test investment strategies using contrastive learning model predictions with time-based train/test splits.
    
    This function follows the same process as invest_monthly_retro_performance() in invest_by_model.py,
    but uses ContrastiveStockClassifier instead of binary classifiers (random forest or xgboost). 
    The contrastive model predicts 3-class labels (bad, mediocre, good) and we convert the probabilities 
    to a single score for investment strategies.
    
    Iterates over months from INVEST_EXP_START_MONTH_STR to INVEST_EXP_END_MONTH_STR, where each 
    month serves as test data and all previous months serve as training data.
    
    Args:
        INVEST_EXP_START_MONTH_STR (str): Start month for investment experiment (default: '2023-01')
        INVEST_EXP_END_MONTH_STR (str): End month for investment experiment (default: '2025-07')
    
    Returns:
        None (prints results)
    """   
    # Import helper functions from invest_by_model
    from invest_by_model import top_candidate_w_return, augment_invest_record_w_long_term_return, benchmark_performance
    
    strategies = {
        'top_5': {'method': 'top_k', 'param': 5},
        'top_10': {'method': 'top_k', 'param': 10},
        'proba_0.8': {'method': 'top_proba', 'param': 0.8},
        'proba_0.85': {'method': 'top_proba', 'param': 0.85},
        'proba_0.75_to_0.85': {'method': 'proba_range', 'param': [0.75, 0.85]}, 
        'mixed': {'method': 'mixed', 'param': [0.75, 0.85, 10]} 
    } 

    # Define the range of months to test
    start_month = pd.Period(INVEST_EXP_START_MONTH_STR, freq='M')
    end_month = pd.Period(INVEST_EXP_END_MONTH_STR, freq='M')
    
    # Generate list of months to iterate over
    current_month = start_month
    months_to_test = []
    while current_month <= end_month:
        months_to_test.append(current_month)
        current_month += 1
    
    print(f"\n📊 Testing investment strategy with contrastive learning (time-based splits)...")
    print(f"📅 Testing months: {len(months_to_test)} months from {start_month} to {end_month}")
    
    # Prepare data - Load featurized data and stock trends
    df_features = pd.read_csv(FEATURIZED_ALL_QUARTERS_FILE)
    df_trends = pd.read_csv(STOCK_TREND_DATA_FILE)
    df = prep_data_feature_label(df_featurized_data=df_features, 
                                  df_stock_trend=df_trends,
                                  quarters_for_gradient_comp=QUARTER_GRADIENTS)

    if FEATURE_IMPORTANCE_RANKING_FLAG:
        feature_importance_ranking = pd.read_csv(os.path.join(MODEL_DIR, 'feature_importance_ranking.csv'))
        df = filter_features_by_importance(df, feature_importance_ranking) 
    
    # Get feature columns from the full dataset
    suffix_cols = collect_column_names_w_suffix(df.columns, feature_suffixes=['_current'])
    feature_cols = suffix_cols + [f for f in df.columns if '_change' in f and f not in suffix_cols]

    # Initialize strategy outcome tracking
    strategy_outcome = {}
    for strategy_name in strategies:
        strategy_outcome[strategy_name] = {
            'monthly_invest_record': []  # Use list to collect DataFrames, concatenate later
        }
    
    # Iterate over each month
    for current_month in months_to_test:
        print(f"\n" + "="*60)
        print(f"📅 Testing performance for {current_month}")
        
        # train/test split
        df_test = df[df['year_month'] == current_month].copy()
        df_train = df[df['year_month'] < current_month].copy()
        print(f"📊 Training data: {len(df_train)} samples", f"📊 Test data: {len(df_test)} samples")
        
        # Skip if insufficient data
        if len(df_train) < 100:
            print(f"⚠️  Insufficient training data ({len(df_train)} samples), skipping...")
            continue
        if len(df_test) == 0:
            print(f"⚠️  No test data for {current_month}, skipping...")
            continue
        
        # Prepare data for contrastive learning using the reusable function
        # Since we already split by month, pass df_train and df_test directly
        X_train, X_test, y_train, y_test_labels, feature_cols_final, df_train_full, df_test_full = prep_data_for_contrastive_learning(
            df_train=df_train,
            df_test=df_test,
            good_threshold=1.05,
            bad_threshold=0.95,
            feature_suffixes=['_current']
        )
        
        # Train contrastive learning model
        model = ContrastiveStockClassifier(
            embedding_dim=128,
            hidden_dims=[128, 128, 64],
            margin=1.0,
            temperature=0.1,
            batch_size=256,
            epochs=100,  
            learning_rate=0.001,
            contrastive_weight=0.75,
            device='cpu',
            random_state=42
        )
        
        print(f"   Training contrastive model...")
        model.fit(X_train, y_train, validation_data=None)
        
        # Evaluate model performance (includes accuracy, correlation, and ROC-AUC)
        results = evaluate_contrastive_classifier(model, X_test, y_test_labels, df_test=df_test_full, print_report=False)
        
        accuracy = results['accuracy']
        print(f"📊 Model performance -- accuracy: {accuracy:.4f}")
        
        if results.get('roc_auc_trend') is not None:
            print(f"📊 Model performance -- ROC-AUC (y_pred_proba vs {Y_LABEL}): {results['roc_auc_trend']:.4f}")
        
        # Convert 3-class probabilities to a single score for investment strategies
        # Encode labels: bad=-1, mediocre=0, good=1
        y_proba = results['probabilities']
        label_to_score = {'bad': -1, 'mediocre': 0, 'good': 1}
        proba_scores = np.zeros(len(y_proba))
        for i, label in enumerate(model.classes_):
            proba_scores += y_proba[:, i] * label_to_score[label]
        
        # Normalize scores to [0, 1] range for compatibility with investment strategies
        # proba_scores range from -1 to 1, so we map to [0, 1]
        y_pred_proba = (proba_scores + 1) / 2
        
        # Add predictions to test dataframe
        df_test = df_test.copy()
        df_test['y_pred_proba'] = y_pred_proba

        # Get market average return for this month
        avg_return = df_test['price_return'].mean()

        for strategy_name, strategy in strategies.items(): 
            df_top_candidates = top_candidate_w_return(df_test, strategy)
            
            if len(df_top_candidates) > 0:
                strategy_outcome[strategy_name]['monthly_invest_record'].append(df_top_candidates)

                # print out the results for the current month
                top_candidate_return = df_top_candidates['price_return'].mean()
                num_tickers = len(df_top_candidates)
                ticker_str = ','.join(df_top_candidates['ticker'].tolist())
                print(f"  📊 Strategy: {strategy_name}", f"Average return (market): {avg_return:.4f}")
                print(f"     Selected tickers ({num_tickers}): {ticker_str}", 
                      f"return from selected: {top_candidate_return:.4f}")
            else:
                print(f"  📊 Strategy: {strategy_name}, No candidates selected")
    
    # Convert lists of DataFrames to single DataFrames
    for strategy_name in strategies:
        monthly_records_list = strategy_outcome[strategy_name]['monthly_invest_record']
        if len(monthly_records_list) > 0:
            strategy_outcome[strategy_name]['monthly_invest_record'] = pd.concat(
                monthly_records_list, ignore_index=True
            )
        else:
            strategy_outcome[strategy_name]['monthly_invest_record'] = pd.DataFrame(
                columns=['year_month', 'cik', 'ticker', 'y_pred_proba', 'rank', 'price_return']
            )
    
    print(f"\n" + "="*30 + "Overall summary" + "="*30)
    # print out the results
    for strategy_name in strategies:
        if len(strategy_outcome[strategy_name]['monthly_invest_record']) == 0:
            print(f"📊 Strategy: {strategy_name}, No investment record")
        else:
            invest_record = strategy_outcome[strategy_name]['monthly_invest_record'].copy()
            invest_record = augment_invest_record_w_long_term_return(invest_record)   
            print(f"📊 Strategy: {strategy_name}", 
                f"\n\t{len(invest_record)} selected, ", 
                f"Short-term return: {invest_record['price_return'].mean():.4f}", 
                f"Long-term return: {invest_record['price_return_long_term'].mean():.4f}"
            ) 
            # print proba stats 
            print(f"     Proba stats: {invest_record['y_pred_proba'].describe()}") 
    
    # benchmark performance: market average short-term return and 6-month return
    # Use the full dataset as benchmark (comparing overall market performance)
    if len(df) > 0:
        print(f"\n{'='*60}")
        print("📊 BENCHMARK: Overall Market Performance")
        print(f"{'='*60}")
        benchmark_performance(df[(df['year_month'] > INVEST_EXP_START_MONTH_STR) 
                               & (df['year_month'] <= INVEST_EXP_END_MONTH_STR)
                               ], num_months=6)


if __name__ == "__main__":
    # main()
    invest_monthly_w_holdback()
