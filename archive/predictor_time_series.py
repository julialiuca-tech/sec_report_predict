#!/usr/bin/env python3
"""
Time Series Predictor for Stock Price Forecasting

This module provides time series analysis methods to predict stock prices using quarterly
financial features from SEC filings. The module supports multiple time series approaches:

1. Baseline Methods:
   - Moving Average (MA)
   - Exponential Smoothing
   - Linear Trend

2. Machine Learning with Temporal Features:
   - XGBoost/LightGBM with lag features
   - Feature aggregation over rolling windows

3. Deep Learning Sequence Models:
   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Unit)
   - Transformer-based models (Time Series Transformer)

Key Challenge: Features are quarterly (period column) while targets are monthly (month_end_date).
This module handles the temporal alignment between these different frequencies.

"""


import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Try importing deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from config import FEATURIZED_ALL_QUARTERS_FILE, STOCK_TREND_DATA_FILE, MODEL_DIR
from utility_data import filter_companies_by_criteria


# =============================================================================
# DATA PREPARATION AND ALIGNMENT
# =============================================================================

def align_quarterly_features_to_monthly_targets(df_features, df_trends):
    """
    Align quarterly feature data with monthly target data.
    
    For each month in df_trends, find the most recent quarterly feature observation
    that occurred before or on that month.
    
    Args:
        df_features: DataFrame with columns (cik, period, data_qtr, feature_0, feature_1, ...)
                   where 'period' is a date (quarterly observations)
        df_trends: DataFrame with columns (cik, ticker, month_end_date, ..., future_close_price)
                  where 'month_end_date' is monthly
    
    Returns:
        pd.DataFrame: Merged DataFrame with features aligned to monthly targets
    """
    # Convert period to datetime if it's not already
    df_features = df_features.copy()
    df_trends = df_trends.copy()
    
    if df_features['period'].dtype != 'datetime64[ns]':
        # Assume period is YYYYMMDD format
        df_features['period'] = pd.to_datetime(df_features['period'], format='%Y%m%d', errors='coerce')
    
    if df_trends['month_end_date'].dtype != 'datetime64[ns]':
        df_trends['month_end_date'] = pd.to_datetime(df_trends['month_end_date'])
    
    # Remove rows with invalid dates
    df_features = df_features[df_features['period'].notna()].copy()
    df_trends = df_trends[df_trends['month_end_date'].notna()].copy()
    
    # Sort both dataframes - critical for merge_asof performance
    # Must be sorted by 'by' column (cik) first, then by merge key
    df_features = df_features.sort_values(['cik', 'period']).reset_index(drop=True)
    df_trends = df_trends.sort_values(['cik', 'month_end_date']).reset_index(drop=True)
    
    # Use merge_asof for efficient vectorized alignment
    # This is much faster than nested loops
    try:
        df_merged = pd.merge_asof(
            df_trends,
            df_features,
            left_on='month_end_date',
            right_on='period',
            by='cik',
            direction='backward',  # Take the most recent observation <= month_end_date
            suffixes=('', '_feature')
        )
        
        # Remove rows where no matching feature was found (period will be NaN)
        df_merged = df_merged[df_merged['period'].notna()].copy()
        
        return df_merged
    
    except (ValueError, KeyError) as e:
        # Fallback to grouped apply if merge_asof fails (e.g., sorting issues)
        # This is still faster than nested loops because it uses vectorized operations within groups
        def align_group(group):
            """Align features to trends for a single cik group."""
            cik_val = group.name
            df_cik_trends = group
            df_cik_features = df_features[df_features['cik'] == cik_val].sort_values('period')
            
            if len(df_cik_features) == 0:
                return pd.DataFrame()
            
            # Use merge_asof on this single group (should work since it's pre-sorted)
            try:
                aligned = pd.merge_asof(
                    df_cik_trends.sort_values('month_end_date'),
                    df_cik_features,
                    left_on='month_end_date',
                    right_on='period',
                    direction='backward',
                    suffixes=('', '_feature')
                )
                return aligned[aligned['period'].notna()]
            except (ValueError, KeyError):
                # Last resort: use vectorized operations
                result_rows = []
                for _, trend_row in df_cik_trends.iterrows():
                    month_date = trend_row['month_end_date']
                    # Vectorized filtering
                    valid_features = df_cik_features[df_cik_features['period'] <= month_date]
                    if len(valid_features) > 0:
                        latest = valid_features.iloc[-1]
                        merged_row = {**latest.to_dict(), **trend_row.to_dict()}
                        result_rows.append(merged_row)
                return pd.DataFrame(result_rows) if result_rows else pd.DataFrame()
        
        # Apply alignment function to each cik group
        df_merged = df_trends.groupby('cik', group_keys=False).apply(align_group).reset_index(drop=True)
        
        return df_merged


def prepare_time_series_data(df_features, df_trends, target_col='future_close_price',
                             feature_cols=None, min_sequence_length=4, max_sequence_length=20):
    """
    Prepare data for time series modeling by creating sequences for each company.
    
    Args:
        df_features: Quarterly feature DataFrame
        df_trends: Monthly target DataFrame
        target_col: Column name for target variable (default: 'future_close_price')
        feature_cols: List of feature column names (if None, auto-detect)
        min_sequence_length: Minimum number of time steps required per company
        max_sequence_length: Maximum number of time steps to use per company
    
    Returns:
        dict: Dictionary with keys:
            - 'sequences': List of feature sequences (each is numpy array of shape [T, F])
            - 'targets': List of target values (each is numpy array of shape [T])
            - 'ciks': List of CIK values corresponding to each sequence
            - 'feature_names': List of feature column names
            - 'scaler': Fitted StandardScaler for features
    """
    # Align quarterly features to monthly targets
    df_aligned = align_quarterly_features_to_monthly_targets(df_features, df_trends)
    
    if len(df_aligned) == 0:
        raise ValueError("No aligned data found after merging features and trends")
    
    # Identify feature columns (exclude metadata columns)
    if feature_cols is None:
        metadata_cols = ['cik', 'period', 'data_qtr', 'ticker', 'month_end_date', 
                        'trend_up_or_down', 'trend_5per_up', 'price_return', 
                        'close_price', 'future_close_price']
        feature_cols = [col for col in df_aligned.columns if col not in metadata_cols]
    
    # Filter out rows with missing targets
    df_aligned = df_aligned[df_aligned[target_col].notna()].copy()
    
    # Sort by cik and month_end_date
    df_aligned = df_aligned.sort_values(['cik', 'month_end_date'])
    
    # Create sequences for each company
    sequences = []
    targets = []
    ciks = []
    trend_up_or_down_list = []  # Store trend_up_or_down for ROC-AUC calculation
    
    for cik in df_aligned['cik'].unique():
        df_cik = df_aligned[df_aligned['cik'] == cik].copy()
        
        # Extract feature matrix for this company
        X_cik = df_cik[feature_cols].values
        
        # Handle NaN values: forward fill, then backward fill, then zero fill
        X_cik_df = pd.DataFrame(X_cik)
        X_cik_df = X_cik_df.ffill().bfill().fillna(0)
        X_cik = X_cik_df.values
        
        # Extract targets
        y_cik = df_cik[target_col].values
        
        # Extract trend_up_or_down if available (for ROC-AUC)
        has_trend_label = 'trend_up_or_down' in df_cik.columns
        if has_trend_label:
            trend_cik = df_cik['trend_up_or_down'].values
        else:
            trend_cik = None
        
        # Filter out rows with all NaN features
        valid_mask = ~np.isnan(X_cik).all(axis=1)
        X_cik = X_cik[valid_mask]
        y_cik = y_cik[valid_mask]
        if trend_cik is not None:
            trend_cik = trend_cik[valid_mask]
        
        # Apply sequence length constraints
        if len(X_cik) < min_sequence_length:
            continue
        
        if len(X_cik) > max_sequence_length:
            # Take the most recent max_sequence_length observations
            X_cik = X_cik[-max_sequence_length:]
            y_cik = y_cik[-max_sequence_length:]
            if trend_cik is not None:
                trend_cik = trend_cik[-max_sequence_length:]
        
        sequences.append(X_cik)
        targets.append(y_cik)
        ciks.append(cik)
        
        # Store the last trend_up_or_down value for this sequence (corresponds to prediction target)
        if trend_cik is not None and len(trend_cik) > 0:
            trend_up_or_down_list.append(trend_cik[-1])
        else:
            trend_up_or_down_list.append(np.nan)
    
    if len(sequences) == 0:
        raise ValueError("No valid sequences found after filtering")
    
    # Fit scaler on all sequences (for standardization)
    all_features = np.concatenate(sequences, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Scale all sequences
    sequences_scaled = [scaler.transform(seq) for seq in sequences]
    
    return {
        'sequences': sequences_scaled,
        'targets': targets,
        'ciks': ciks,
        'feature_names': feature_cols,
        'scaler': scaler,
        'trend_up_or_down': trend_up_or_down_list  # For ROC-AUC calculation
    }


# =============================================================================
# BASELINE TIME SERIES METHODS
# =============================================================================

class MovingAveragePredictor:
    """Simple moving average baseline predictor."""
    
    def __init__(self, window_size=4):
        self.window_size = window_size
        self.targets_ = None
    
    def fit(self, sequences, targets):
        """Fit the model (baseline doesn't need training)."""
        self.targets_ = targets
        return self
    
    def predict(self, sequences):
        """Predict using moving average of historical prices."""
        predictions = []
        for seq, target_seq in zip(sequences, self.targets_):
            # For moving average, we need historical prices
            # Since we only have features, we'll use a simple average of the last window_size observations
            # This is a simplified version - in practice, you'd want historical prices
            if len(target_seq) >= self.window_size:
                # Simple approach: predict based on historical target average
                # In a real scenario, you'd use historical prices
                pred = np.mean(target_seq[-self.window_size:])
            else:
                pred = np.mean(target_seq) if len(target_seq) > 0 else 0.0
            predictions.append(pred)
        return np.array(predictions)


# =============================================================================
# MACHINE LEARNING WITH LAG FEATURES
# =============================================================================

def create_lag_features(df_aligned, feature_cols, lags=[1, 2, 4, 8], target_col='future_close_price'):
    """
    Create lag features for time series prediction.
    
    Args:
        df_aligned: DataFrame with aligned features and targets, sorted by (cik, month_end_date)
        feature_cols: List of feature column names
        lags: List of lag periods to create (in quarters)
        target_col: Target column name
    
    Returns:
        pd.DataFrame: DataFrame with lag features added
    """
    df = df_aligned.copy()
    df = df.sort_values(['cik', 'month_end_date'])
    
    # Group by cik and create lag features
    lag_features = []
    for cik in df['cik'].unique():
        df_cik = df[df['cik'] == cik].copy().reset_index(drop=True)
        
        for lag in lags:
            for col in feature_cols:
                lag_col_name = f'{col}_lag_{lag}'
                df_cik[lag_col_name] = df_cik[col].shift(lag)
        
        lag_features.append(df_cik)
    
    df_with_lags = pd.concat(lag_features, ignore_index=True)
    
    # Drop rows with NaN in lag features
    # Strategy: Keep rows where at least the smallest lag (most recent history) is available
    # This ensures we don't drop all data for companies with limited history
    if len(lags) > 0 and len(feature_cols) > 0:
        min_lag = min(lags)
        # Check that at least some features have the minimum lag available
        # Use a sample of features (e.g., first 10 or 50) to avoid being too strict
        sample_size = min(50, len(feature_cols))  # Check up to 50 features
        min_lag_cols = [f'{col}_lag_{min_lag}' for col in feature_cols[:sample_size]]
        # Only keep rows where at least one min_lag feature is not NaN
        # This is much less aggressive than requiring all lag features
        if len(min_lag_cols) > 0:
            valid_rows = df_with_lags[min_lag_cols].notna().any(axis=1)
            df_with_lags = df_with_lags[valid_rows].copy()
    
    return df_with_lags


class LagFeaturePredictor:
    """Predictor using lag features with tree-based models."""
    
    def __init__(self, model_type='xgboost', lags=[1, 2, 4], **model_params):
        self.model_type = model_type
        self.lags = lags
        self.model_params = model_params  # User-provided parameters (may be empty)
        self.model_ = None
        self.feature_cols_ = None
        self.final_params_ = None  # Will store the final merged parameters after fit()
    
    def fit(self, df_aligned, feature_cols, target_col='future_close_price'):
        """Fit the model using lag features."""
        # Validate inputs
        if len(df_aligned) == 0:
            raise ValueError("df_aligned is empty")
        if len(feature_cols) == 0:
            raise ValueError("feature_cols is empty")
        if target_col not in df_aligned.columns:
            raise ValueError(f"Target column '{target_col}' not found in df_aligned")
        
        # Create lag features
        print(f"   Creating lag features with lags={self.lags} from {len(df_aligned)} rows...")
        df_with_lags = create_lag_features(df_aligned, feature_cols, self.lags, target_col)
        print(f"   After creating lag features: {len(df_with_lags)} rows")
        
        if len(df_with_lags) == 0:
            raise ValueError(f"No data remaining after creating lag features. "
                           f"Input had {len(df_aligned)} rows. "
                           f"Try using smaller lag values or ensure companies have sufficient history.")
        
        # Prepare features and target - get lag column names that actually exist
        lag_col_names = [col for col in df_with_lags.columns if any(col.endswith(f'_lag_{lag}') for lag in self.lags)]
        
        if len(lag_col_names) == 0:
            raise ValueError(f"No lag features found. Expected columns ending with '_lag_1', '_lag_2', etc.")
        
        X = df_with_lags[lag_col_names].fillna(0)
        y = df_with_lags[target_col]
        
        # Filter out NaN targets
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid data remaining after filtering NaN targets")
        
        if len(X.columns) == 0:
            raise ValueError("No feature columns available for training")
        
        self.feature_cols_ = X.columns.tolist()
        
        # Initialize and train model
        if self.model_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed")
            default_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}
            default_params.update(self.model_params)
            self.final_params_ = default_params.copy()  # Store final merged parameters
            self.model_ = xgb.XGBRegressor(**default_params)
        elif self.model_type == 'lightgbm':
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM not installed")
            default_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'verbosity': -1}
            default_params.update(self.model_params)
            self.final_params_ = default_params.copy()  # Store final merged parameters
            self.model_ = lgb.LGBMRegressor(**default_params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        # Ensure final_params_ is set (should have been set above, but just in case)
        if not hasattr(self, 'final_params_') or self.final_params_ is None:
            self.final_params_ = {}
        
        # Train the model
        try:
            self.model_.fit(X, y)
            print(f"   ✅ Model trained successfully: {len(X)} samples, {len(X.columns)} features")
        except Exception as e:
            print(f"   ❌ Model training failed: {str(e)}")
            print(f"      X shape: {X.shape}, y shape: {y.shape}")
            raise
        
        return self
    
    def predict(self, df_aligned, feature_cols):
        """Predict using lag features."""
        df_with_lags = create_lag_features(df_aligned, feature_cols, self.lags)
        lag_col_names = [col for col in df_with_lags.columns if any(col.endswith(f'_lag_{lag}') for lag in self.lags)]
        
        # Only use columns that were in training
        available_cols = [col for col in lag_col_names if col in self.feature_cols_]
        if len(available_cols) == 0:
            raise ValueError("No matching lag feature columns found")
        
        X = df_with_lags[available_cols].fillna(0)
        
        # Reorder columns to match training
        X = X[[col for col in self.feature_cols_ if col in X.columns]]
        
        return self.model_.predict(X)


# =============================================================================
# DEEP LEARNING SEQUENCE MODELS
# =============================================================================

if HAS_TORCH:
    def pad_collate_fn(batch):
        """
        Custom collate function to pad variable-length sequences to the same length.
        
        Args:
            batch: List of tuples (sequence, target) where sequences have different lengths
        
        Returns:
            tuple: (padded_sequences, targets, lengths) where:
                - padded_sequences: Tensor of shape (batch_size, max_len, feature_dim) with padding
                - targets: Tensor of shape (batch_size, 1)
                - lengths: Tensor of original sequence lengths (for pack_padded_sequence if needed)
        """
        sequences, targets = zip(*batch)
        
        # Get sequence lengths
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        max_len = lengths.max().item()
        
        # Get feature dimension
        feature_dim = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
        
        # Pad sequences to max length
        padded_sequences = []
        for seq in sequences:
            seq_tensor = torch.FloatTensor(seq)
            if len(seq_tensor.shape) == 1:
                seq_tensor = seq_tensor.unsqueeze(1)
            
            # Pad with zeros at the beginning (so the last timestep is always the actual last observation)
            pad_length = max_len - len(seq_tensor)
            if pad_length > 0:
                padding = torch.zeros(pad_length, feature_dim)
                seq_tensor = torch.cat([padding, seq_tensor], dim=0)
            
            padded_sequences.append(seq_tensor)
        
        # Stack into batch tensor
        padded_batch = torch.stack(padded_sequences)
        
        # Stack targets
        target_batch = torch.stack([torch.FloatTensor([t]) for t in targets])
        
        return padded_batch, target_batch, lengths
    
    class TimeSeriesDataset(Dataset):
        """PyTorch Dataset for time series sequences."""
        
        def __init__(self, sequences, targets):
            self.sequences = sequences
            self.targets = targets
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            # Return last target in sequence (predicting future from past)
            return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx][-1]])
    
    
    class LSTMPredictor(nn.Module):
        """LSTM-based time series predictor."""
        
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(LSTMPredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # Use last output
            last_output = lstm_out[:, -1, :]
            output = self.fc(last_output)
            return output
    
    
    class GRUPredictor(nn.Module):
        """GRU-based time series predictor."""
        
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(GRUPredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.gru = nn.GRU(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            gru_out, _ = self.gru(x)
            # Use last output
            last_output = gru_out[:, -1, :]
            output = self.fc(last_output)
            return output


def train_lstm_model(sequences_train, targets_train, sequences_val, targets_val,
                    input_size, hidden_size=64, num_layers=2, epochs=50, 
                    batch_size=32, learning_rate=0.001, device='cpu', 
                    patience=20, min_delta=0.0):
    """
    Train an LSTM model for time series prediction with early stopping.
    
    Args:
        sequences_train: Training sequences
        targets_train: Training targets
        sequences_val: Validation sequences
        targets_val: Validation targets
        input_size: Input feature size
        hidden_size: Hidden layer size
        num_layers: Number of LSTM layers
        epochs: Maximum number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        patience: Number of epochs to wait before early stopping
        min_delta: Minimum change in validation loss to qualify as improvement
    
    Returns:
        Trained model (best model based on validation loss)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not installed")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(sequences_train, targets_train)
    val_dataset = TimeSeriesDataset(sequences_val, targets_val)
    
    # Use custom collate function to handle variable-length sequences
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    # Initialize model
    model = LSTMPredictor(input_size, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences, targets, lengths in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets, lengths in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience}). Best val loss: {best_val_loss:.6f}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    return model


def train_gru_model(sequences_train, targets_train, sequences_val, targets_val,
                   input_size, hidden_size=64, num_layers=2, epochs=50,
                   batch_size=32, learning_rate=0.001, device='cpu',
                   patience=20, min_delta=0.0):
    """
    Train a GRU model for time series prediction with early stopping.
    
    Args:
        sequences_train: Training sequences
        targets_train: Training targets
        sequences_val: Validation sequences
        targets_val: Validation targets
        input_size: Input feature size
        hidden_size: Hidden layer size
        num_layers: Number of GRU layers
        epochs: Maximum number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        patience: Number of epochs to wait before early stopping
        min_delta: Minimum change in validation loss to qualify as improvement
    
    Returns:
        Trained model (best model based on validation loss)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not installed")
    
    # Similar to LSTM training
    train_dataset = TimeSeriesDataset(sequences_train, targets_train)
    val_dataset = TimeSeriesDataset(sequences_val, targets_val)
    
    # Use custom collate function to handle variable-length sequences
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    model = GRUPredictor(input_size, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for sequences, targets, lengths in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets, lengths in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience}). Best val loss: {best_val_loss:.6f}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    return model


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_predictions(y_true, y_pred, df_test=None, print_report=True):
    """
    Evaluate regression predictions.
    
    Args:
        y_true: True target values (future_close_price)
        y_pred: Predicted target values
        df_test: Optional DataFrame with 'trend_up_or_down' column for ROC-AUC calculation.
                 Should have same number of rows as original y_true/y_pred before NaN filtering.
        print_report: Whether to print evaluation report
    
    Returns:
        dict: Dictionary with evaluation metrics including ROC-AUC if df_test provided
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Store original length for index alignment with df_test
    original_length = len(y_true)
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_filtered = y_true[valid_mask]
    y_pred_filtered = y_pred[valid_mask]
    
    if len(y_true_filtered) == 0:
        if print_report:
            print("\n⚠️  WARNING: No valid data after filtering NaN values. Cannot evaluate predictions.")
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'correlation': np.nan, 'roc_auc': np.nan}
    
    y_true = y_true_filtered
    y_pred = y_pred_filtered
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    try:
        correlation, p_value = pearsonr(y_true, y_pred)
    except:
        correlation = np.nan
    
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation
    }
    
    # Compute ROC-AUC using predicted price as scores and trend_up_or_down as truth
    roc_auc = None
    if df_test is not None and 'trend_up_or_down' in df_test.columns:
        # Apply the same valid_mask filtering to df_test
        # valid_mask corresponds to the original indices
        assert isinstance(df_test, pd.DataFrame), "df_test must be a pandas DataFrame"
        assert len(df_test) == original_length, f"df_test length ({len(df_test)}) must match original_length ({original_length})"
        valid_indices = np.where(valid_mask)[0]
        df_test_filtered = df_test.iloc[valid_indices]
        y_label_values = df_test_filtered['trend_up_or_down'].values
        
        # Remove NaN values for ROC-AUC
        roc_valid_mask = ~np.isnan(y_label_values)
        if roc_valid_mask.sum() > 0:
            y_label_binary = y_label_values[roc_valid_mask]
            y_pred_scores = y_pred[roc_valid_mask]
            
            # Ensure binary labels (should be 0/1)
            unique_labels = np.unique(y_label_binary)
            if len(unique_labels) == 2 and set(unique_labels).issubset({0, 1}):
                try:
                    roc_auc = roc_auc_score(y_label_binary, y_pred_scores)
                    results['roc_auc'] = roc_auc
                except ValueError:
                    roc_auc = None
                    results['roc_auc'] = None
    
    if print_report:
        print("\n" + "="*60)
        print("PREDICTION EVALUATION")
        print("="*60)
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  Correlation: {correlation:.4f}")
        if roc_auc is not None:
            print(f"  ROC-AUC (predicted price vs trend_up_or_down): {roc_auc:.4f}")
        elif df_test is not None and 'trend_up_or_down' in df_test.columns:
            print(f"  ROC-AUC (predicted price vs trend_up_or_down): Not available (insufficient data or non-binary labels)")
        print("="*60)
    
    return results


# =============================================================================
# HELPER FUNCTIONS FOR MODEL LOADING AND EVALUATION
# =============================================================================

def load_and_evaluate_lag_model(model_path, df_test, feature_cols, model_name="Model"):
    """
    Load a LagFeaturePredictor (XGBoost or LightGBM) from pickle and evaluate it.
    
    Args:
        model_path: Path to the pickle file
        df_test: Test DataFrame
        feature_cols: Feature column names
        model_name: Name for logging
    
    Returns:
        bool: True if successfully loaded and evaluated, False otherwise
    """
    try:
        with open(model_path, 'rb') as f:
            predictor = pickle.load(f)
        
        print(f"   📦 Loaded {model_name} model from: {model_path}")
        
        y_pred = predictor.predict(df_test, feature_cols)
        y_true = df_test['future_close_price'].values[:len(y_pred)]
        
        evaluate_predictions(y_true, y_pred, df_test=df_test)
        return True
    except Exception as e:
        print(f"   ⚠️  Failed to load/evaluate {model_name} model: {str(e)}")
        return False


def load_and_evaluate_sequence_model(model_path, df_features, df_trends, df_test, 
                                     feature_cols, model_name="Model", device=None):
    """
    Load a sequence model (LSTM or GRU) from pickle and evaluate it.
    
    Args:
        model_path: Path to the pickle file
        df_features: Features DataFrame
        df_trends: Trends DataFrame
        df_test: Test DataFrame
        feature_cols: Feature column names
        model_name: Name for logging
        device: torch device (if None, auto-detect)
    
    Returns:
        bool: True if successfully loaded and evaluated, False otherwise
    """
    if not HAS_TORCH:
        print(f"   ⚠️  PyTorch not available, cannot load {model_name} model")
        return False
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        print(f"   📦 Loaded {model_name} model from: {model_path}")
        
        # Reconstruct model architecture
        input_size = model_dict['input_size']
        hidden_size = model_dict['hidden_size']
        num_layers = model_dict['num_layers']
        model_type = model_dict['model_type']
        
        if model_type == 'lstm':
            model = LSTMPredictor(input_size, hidden_size, num_layers).to(device)
        elif model_type == 'gru':
            model = GRUPredictor(input_size, hidden_size, num_layers).to(device)
        else:
            raise ValueError(f"Unknown model type in pickle: {model_type}")
        
        # Load state dict
        model.load_state_dict(model_dict['model_state_dict'])
        model.eval()
        
        # Get scaler (should be saved with model)
        scaler = model_dict.get('scaler')
        if scaler is None:
            print(f"   ⚠️  Warning: Scaler not found in saved model, using test data scaler (may affect results)")
        
        # Prepare test sequences
        test_feature_cols = feature_cols[:input_size]  # Use same number of features as model
        test_sequences, test_targets_list = prepare_test_sequences_for_evaluation(
            df_features, df_trends, df_test, test_feature_cols, scaler
        )
        
        assert len(test_sequences) > 0, f"No test sequences available for {model_name}. Check data preparation."
        
        # If scaler was None, fit a temporary one (not ideal but necessary)
        if scaler is None:
            all_test_features = np.concatenate([seq for seq in test_sequences], axis=0)
            temp_scaler = StandardScaler()
            temp_scaler.fit(all_test_features)
            test_sequences = [temp_scaler.transform(seq) for seq in test_sequences]
        
        # Evaluate
        y_pred, y_true = evaluate_sequence_model(
            model, test_sequences, test_targets_list, device, model_name
        )
        
        evaluate_predictions(y_true, y_pred, df_test=None)
        return True
            
    except Exception as e:
        print(f"   ⚠️  Failed to load/evaluate {model_name} model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# HELPER FUNCTIONS FOR SEQUENCE MODELS
# =============================================================================

def prepare_test_sequences_for_evaluation(df_features, df_trends, df_test, feature_cols, scaler=None):
    """
    Prepare test sequences using the training scaler.
    
    Args:
        df_features: Full features DataFrame
        df_trends: Full trends DataFrame
        df_test: Test DataFrame (for filtering CIKs)
        feature_cols: Feature column names to use
        scaler: Fitted StandardScaler from training
    
    Returns:
        tuple: (test_sequences, test_targets) where each is a list
    """
    # Align quarterly features to monthly targets for test data
    df_aligned_test = align_quarterly_features_to_monthly_targets(
        df_features[df_features['cik'].isin(df_test['cik'].unique())],
        df_trends[df_trends['cik'].isin(df_test['cik'].unique())]
    )
    
    # Create test sequences with the same scaler
    df_aligned_test = df_aligned_test[df_aligned_test['future_close_price'].notna()].copy()
    df_aligned_test = df_aligned_test.sort_values(['cik', 'month_end_date'])
    
    test_sequences = []
    test_targets = []
    
    for cik in df_aligned_test['cik'].unique():
        df_cik = df_aligned_test[df_aligned_test['cik'] == cik].copy()
        X_cik = df_cik[feature_cols].values
        X_cik_df = pd.DataFrame(X_cik)
        X_cik_df = X_cik_df.ffill().bfill().fillna(0)
        X_cik = X_cik_df.values
        y_cik = df_cik['future_close_price'].values
        
        valid_mask = ~np.isnan(X_cik).all(axis=1)
        X_cik = X_cik[valid_mask]
        y_cik = y_cik[valid_mask]
        
        if len(X_cik) >= 1:  # At least 1 observation
            if len(X_cik) > 20:  # Truncate to max_sequence_length
                X_cik = X_cik[-20:]
                y_cik = y_cik[-20:]
            
            # Use training scaler to transform test sequences (if provided)
            if scaler is not None:
                X_cik_scaled = scaler.transform(X_cik)
                test_sequences.append(X_cik_scaled)
            else:
                test_sequences.append(X_cik)
            test_targets.append(y_cik)
    
    return test_sequences, test_targets


def evaluate_sequence_model(model, test_sequences, test_targets, device, model_name="Model"):
    """
    Evaluate a sequence model on test data.
    
    Args:
        model: Trained PyTorch model (LSTM or GRU)
        test_sequences: List of test sequences (already scaled)
        test_targets: List of target arrays (each array has multiple time steps)
        device: torch device
        model_name: Name for logging
    
    Returns:
        tuple: (y_pred, y_true) as numpy arrays
    """
    model.eval()
    test_predictions = []
    test_targets_flat = []
    
    with torch.no_grad():
        for i, seq in enumerate(test_sequences):
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)  # Add batch dimension
            pred = model(seq_tensor)
            test_predictions.append(pred.cpu().item())
            test_targets_flat.append(test_targets[i][-1])  # Last target value
    
    return np.array(test_predictions), np.array(test_targets_flat)


def train_and_evaluate_sequence_model(model_type, df_features, df_trends, df_train, df_test, 
                                     feature_cols, feature_subset_size=100,
                                     hidden_size=64, num_layers=2, epochs=30, 
                                     batch_size=32, device=None):
    """
    Train, save, and evaluate a sequence model (LSTM or GRU).
    
    Args:
        model_type: 'lstm' or 'gru'
        df_features: Features DataFrame
        df_trends: Trends DataFrame
        df_train: Training DataFrame (for filtering CIKs)
        df_test: Test DataFrame (for filtering CIKs)
        feature_cols: List of all feature column names
        feature_subset_size: Number of features to use (default: 100)
        hidden_size: Hidden size for the model
        num_layers: Number of layers
        epochs: Number of training epochs
        batch_size: Batch size
        device: torch device (if None, auto-detect)
    
    Returns:
        tuple: (trained_model, ts_data) or (None, None) if training failed
    """
    if not HAS_TORCH:
        print(f"   ⚠️  PyTorch not available, skipping {model_type.upper()}")
        return None, None
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_name = model_type.upper()
    
    try:
        # Prepare sequence data
        ts_data = prepare_time_series_data(
            df_features[df_features['cik'].isin(df_train['cik'].unique())],
            df_trends[df_trends['cik'].isin(df_train['cik'].unique())],
            feature_cols=feature_cols[:feature_subset_size]
        )
        
        # Split sequences into train/val
        n_train = int(0.8 * len(ts_data['sequences']))
        sequences_train = ts_data['sequences'][:n_train]
        targets_train = ts_data['targets'][:n_train]
        sequences_val = ts_data['sequences'][n_train:]
        targets_val = ts_data['targets'][n_train:]
        
        input_size = len(ts_data['feature_names'])
        
        print(f"   Training {model_name} on {len(sequences_train)} sequences...")
        print(f"   Using device: {device}")
        
        # Train model
        if model_type == 'lstm':
            model = train_lstm_model(
                sequences_train, targets_train, sequences_val, targets_val,
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                epochs=epochs, batch_size=batch_size, device=device
            )
        elif model_type == 'gru':
            model = train_gru_model(
                sequences_train, targets_train, sequences_val, targets_val,
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                epochs=epochs, batch_size=batch_size, device=device
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        print(f"   ✅ {model_name} training completed")
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f'time_series_{model_type}_predictor.pkl')
        model_save_dict = {
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'model_type': model_type,
            'scaler': ts_data['scaler']  # Save scaler for proper evaluation
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_save_dict, f)
        print(f"   💾 Model saved to: {model_path}")
        
        # Evaluate on test data
        print(f"\n   📊 Evaluating {model_name} on test data...")
        try:
            test_feature_cols = feature_cols[:feature_subset_size]
            test_sequences, test_targets_list = prepare_test_sequences_for_evaluation(
                df_features, df_trends, df_test, test_feature_cols, ts_data['scaler']
            )
            
            y_pred, y_true = evaluate_sequence_model(
                model, test_sequences, test_targets_list, device, model_name
            )
            
            evaluate_predictions(y_true, y_pred, df_test=None)
            
        except Exception as e:
            print(f"   ⚠️  {model_name} evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return model, ts_data
        
    except Exception as e:
        print(f"   ⚠️  {model_name} training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# =============================================================================
# MAIN DEMONSTRATION FUNCTION
# =============================================================================

# Note: The code has several problems, putting on the back burner for now.
# Problems: 
# 1. The code takes awefully long to train a model and also very long for scoring. 
# The performance of XGBoost and LightGBM is inferior to their classifier version. 
# 2. The train/test split strategy is inconsistent among models: 
#   (i) In main() function, for XGBoost and LightGBM, it uses the split strategy 
#     defined in the config.py file (e.g., SPLIT_STRATEGY = {'period':'bottom'}).
#   (ii) for LSTM and GRU, in the function train_and_evaluate_sequence_model(),
#      it uses a 80-20 split among time series sequences for training and validation, 
#      and then uses the test dataset for evaluation. 
#   (iii) yet in the load_and_evaluate_sequence_model() function, it uses df_test to 
#      define the list of ciks, and then use all sequences with those ciks for evaluation. 
#      The logic is in the prepare_time_series_for_evaluation() function.
# 3. The function prepare_time_series_for_evaluation() has a for-loop over all ciks 
#    in df_test and is very slow.
# Among these, 2 is actually a major problem with no straightforward solution.

def main_archived():
    """Demonstration of time series prediction methods."""
    print("="*70)
    print("TIME SERIES STOCK PRICE PREDICTOR")
    print("="*70)
    
    # Load data
    print("\n📊 Loading data...")
    df_features = pd.read_csv(FEATURIZED_ALL_QUARTERS_FILE)
    df_trends = pd.read_csv(STOCK_TREND_DATA_FILE)
    
    print(f"   Features shape: {df_features.shape}")
    print(f"   Trends shape: {df_trends.shape}")
    
    # Filter companies by criteria (single ticker, minimum quarters)
    df_features, df_trends = filter_companies_by_criteria(
        df_features, df_trends, 
        min_quarters=4,  # At least 1 year of data
        remove_multi_ticker=True,
        print_summary=True
    )
    
    # Align data
    print("\n📊 Aligning quarterly features with monthly targets...")
    df_aligned = align_quarterly_features_to_monthly_targets(df_features, df_trends)
    print(f"   Aligned data shape: {df_aligned.shape}")
    
    if len(df_aligned) == 0:
        print("❌ No aligned data found. Check data compatibility.")
        return
    
    # Identify feature columns
    metadata_cols = ['cik', 'period', 'data_qtr', 'ticker', 'month_end_date',
                    'trend_up_or_down', 'trend_5per_up', 'price_return',
                    'close_price', 'future_close_price']
    feature_cols = [col for col in df_aligned.columns if col not in metadata_cols]
    print(f"   Number of features: {len(feature_cols)}")
    
    # Split data by time (use period/month_end_date for splitting)
    if 'month_end_date' in df_aligned.columns:
        df_aligned['month_end_date'] = pd.to_datetime(df_aligned['month_end_date'])
        split_date = df_aligned['month_end_date'].quantile(0.7)
        df_train = df_aligned[df_aligned['month_end_date'] < split_date].copy()
        df_test = df_aligned[df_aligned['month_end_date'] >= split_date].copy()
    else:
        # Fallback: random split
        df_train = df_aligned.sample(frac=0.7, random_state=42)
        df_test = df_aligned.drop(df_train.index)
    
    print(f"   Train samples: {len(df_train)}")
    print(f"   Test samples: {len(df_test)}")
    
    # Test 1: Lag Feature Predictor (XGBoost)
    if HAS_XGBOOST:
        print("\n" + "="*70)
        print("METHOD 1: XGBoost with Lag Features")
        print("="*70)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        xgb_model_path = os.path.join(MODEL_DIR, 'time_series_xgboost_lag_predictor.pkl')
        
        # Check if model exists
        if os.path.exists(xgb_model_path):
            print(f"   📦 Found existing model at: {xgb_model_path}")
            load_and_evaluate_lag_model(xgb_model_path, df_test, feature_cols, "XGBoost")
        else:
            print(f"   🔨 Training new XGBoost model...")
            lag_predictor = LagFeaturePredictor(model_type='xgboost', lags=[1, 2, 4])
            lag_predictor.fit(df_train, feature_cols, target_col='future_close_price')
            
            # Note: model_params contains only user-provided parameters (may be empty)
            # final_params_ contains the merged parameters (defaults + user params) used by the model
            if hasattr(lag_predictor, 'final_params_') and lag_predictor.final_params_ is not None:
                print(f"   Model parameters used: {lag_predictor.final_params_}")
            
            y_pred = lag_predictor.predict(df_test, feature_cols)
            y_true = df_test['future_close_price'].values[:len(y_pred)]
            
            evaluate_predictions(y_true, y_pred, df_test=df_test)
            
            # Save model
            with open(xgb_model_path, 'wb') as f:
                pickle.dump(lag_predictor, f)
            print(f"   💾 Model saved to: {xgb_model_path}")
    
    # Test 2: LightGBM with Lag Features
    if HAS_LIGHTGBM:
        print("\n" + "="*70)
        print("METHOD 2: LightGBM with Lag Features")
        print("="*70)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        lgb_model_path = os.path.join(MODEL_DIR, 'time_series_lightgbm_lag_predictor.pkl')
        
        # Check if model exists
        if os.path.exists(lgb_model_path):
            print(f"   📦 Found existing model at: {lgb_model_path}")
            load_and_evaluate_lag_model(lgb_model_path, df_test, feature_cols, "LightGBM")
        else:
            print(f"   🔨 Training new LightGBM model...")
            lgb_predictor = LagFeaturePredictor(model_type='lightgbm', lags=[1, 2, 4])
            lgb_predictor.fit(df_train, feature_cols, target_col='future_close_price')
            
            # Note: model_params contains only user-provided parameters (may be empty)
            # final_params_ contains the merged parameters (defaults + user params) used by the model
            if hasattr(lgb_predictor, 'final_params_') and lgb_predictor.final_params_ is not None:
                print(f"   Model parameters used: {lgb_predictor.final_params_}")
            
            y_pred = lgb_predictor.predict(df_test, feature_cols)
            y_true = df_test['future_close_price'].values[:len(y_pred)]
            
            evaluate_predictions(y_true, y_pred, df_test=df_test)
            
            # Save model
            with open(lgb_model_path, 'wb') as f:
                pickle.dump(lgb_predictor, f)
            print(f"   💾 Model saved to: {lgb_model_path}")
    
    # Test 3: LSTM (if PyTorch available)
    if HAS_TORCH:
        print("\n" + "="*70)
        print("METHOD 3: LSTM Sequence Model")
        print("="*70)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        lstm_model_path = os.path.join(MODEL_DIR, 'time_series_lstm_predictor.pkl')
        
        # Check if model exists
        if os.path.exists(lstm_model_path):
            print(f"   📦 Found existing model at: {lstm_model_path}")
            load_and_evaluate_sequence_model(
                lstm_model_path, df_features, df_trends, df_test, feature_cols, "LSTM"
            )
        else:
            print(f"   🔨 Training new LSTM model...")
            train_and_evaluate_sequence_model(
                'lstm', df_features, df_trends, df_train, df_test, feature_cols
            )
    
    # Test 4: GRU (if PyTorch available)
    if HAS_TORCH:
        print("\n" + "="*70)
        print("METHOD 4: GRU Sequence Model")
        print("="*70)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        gru_model_path = os.path.join(MODEL_DIR, 'time_series_gru_predictor.pkl')
        
        # Check if model exists
        if os.path.exists(gru_model_path):
            print(f"   📦 Found existing model at: {gru_model_path}")
            load_and_evaluate_sequence_model(
                gru_model_path, df_features, df_trends, df_test, feature_cols, "GRU"
            )
        else:
            print(f"   🔨 Training new GRU model...")
            train_and_evaluate_sequence_model(
                'gru', df_features, df_trends, df_train, df_test, feature_cols
            )
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\n💡 Suggestions for time series methods:")
    print("   1. Lag Feature Models: Good baseline, interpretable")
    print("   2. LSTM/GRU: Can capture long-term dependencies")
    print("   3. Transformer models: State-of-the-art for sequence modeling")
    print("   4. Hybrid approaches: Combine multiple methods")
    print("   5. Feature engineering: Rolling statistics, trends, seasonality")


def main():
    """
    Main function to train and evaluate sequence models (LSTM/GRU) with CIK-based splitting.
    
    This function:
    1. Loads and filters data
    2. Prepares time series sequences for all companies
    3. Splits sequences by CIK values: 70% CIKs for training, 30% for testing
    4. Further splits training sequences: 80% for training, 20% for validation
    5. Trains and evaluates LSTM and GRU models
    """
    if not HAS_TORCH:
        print("⚠️  PyTorch not available. Cannot run sequence models.")
        return
    
    print("="*70)
    print("TIME SERIES STOCK PRICE PREDICTOR (CIK-based Split)")
    print("="*70)
    
    # Load data
    print("\n📊 Loading data...")
    df_features = pd.read_csv(FEATURIZED_ALL_QUARTERS_FILE)
    df_trends = pd.read_csv(STOCK_TREND_DATA_FILE)
    
    print(f"   Features shape: {df_features.shape}")
    print(f"   Trends shape: {df_trends.shape}")
    
    # Filter companies by criteria (single ticker, minimum quarters)
    df_features, df_trends = filter_companies_by_criteria(
        df_features, df_trends, 
        min_quarters=4,  # At least 1 year of data
        remove_multi_ticker=True,
        print_summary=True
    )
    
    # Prepare all time series sequences
    print("\n📊 Preparing time series sequences...")
    
    # Identify feature columns from aligned data (sample alignment to get columns)
    df_aligned_sample = align_quarterly_features_to_monthly_targets(
        df_features.head(1000), df_trends.head(1000)
    )
    metadata_cols = ['cik', 'period', 'data_qtr', 'ticker', 'month_end_date',
                    'trend_up_or_down', 'trend_5per_up', 'price_return',
                    'close_price', 'future_close_price']
    feature_cols = [col for col in df_aligned_sample.columns if col not in metadata_cols]
    print(f"   Number of features: {len(feature_cols)}")
    
    # Prepare sequences for all companies
    ts_data = prepare_time_series_data(
        df_features,
        df_trends,
        feature_cols=feature_cols[:100],  # Use subset for speed
        min_sequence_length=4,
        max_sequence_length=20
    )
    
    print(f"   Prepared {len(ts_data['sequences'])} sequences from {len(ts_data['ciks'])} companies")
    
    # 3-way CIK partition: train/val/test
    # Maintains: 70% (train+val) / 30% test, and 80% train / 20% val within (train+val)
    unique_ciks = np.unique(ts_data['ciks'])
    n_total = len(unique_ciks)
    
    # Calculate split sizes: 56% train, 14% val, 30% test
    # This maintains 70/30 for (train+val)/test and 80/20 for train/val
    n_train_ciks = int(0.56 * n_total)
    n_val_ciks = int(0.14 * n_total)
    # n_test_ciks = n_total - n_train_ciks - n_val_ciks  # Remaining goes to test
    
    # Shuffle CIKs for random split
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    train_cik_indices = indices[:n_train_ciks]
    val_cik_indices = indices[n_train_ciks:n_train_ciks + n_val_ciks]
    test_cik_indices = indices[n_train_ciks + n_val_ciks:]
    
    train_ciks = unique_ciks[train_cik_indices]
    val_ciks = unique_ciks[val_cik_indices]
    test_ciks = unique_ciks[test_cik_indices]
    
    # Split sequences based on CIK membership using DataFrame joins
    # Create DataFrame from ts_data with sequence index and CIK
    df_ts_data = pd.DataFrame({
        'sequence_idx': range(len(ts_data['ciks'])),
        'cik': ts_data['ciks']
    })
    
    # Create DataFrames for train, val, and test CIKs
    df_train_ciks = pd.DataFrame({'cik': train_ciks})
    df_val_ciks = pd.DataFrame({'cik': val_ciks})
    df_test_ciks = pd.DataFrame({'cik': test_ciks})
    
    # Join to get sequence indices for train, val, and test
    df_train = df_ts_data.merge(df_train_ciks, on='cik', how='inner')
    df_val = df_ts_data.merge(df_val_ciks, on='cik', how='inner')
    df_test = df_ts_data.merge(df_test_ciks, on='cik', how='inner')
    
    # Extract sequences and targets using the joined indices
    train_seq_indices = df_train['sequence_idx'].values
    val_seq_indices = df_val['sequence_idx'].values
    test_seq_indices = df_test['sequence_idx'].values
    
    sequences_train = [ts_data['sequences'][i] for i in train_seq_indices]
    targets_train = [ts_data['targets'][i] for i in train_seq_indices]
    ciks_train = df_train['cik'].tolist()
    trend_train = [ts_data['trend_up_or_down'][i] for i in train_seq_indices]
    
    sequences_val = [ts_data['sequences'][i] for i in val_seq_indices]
    targets_val = [ts_data['targets'][i] for i in val_seq_indices]
    ciks_val = df_val['cik'].tolist()
    trend_val = [ts_data['trend_up_or_down'][i] for i in val_seq_indices]
    
    sequences_test = [ts_data['sequences'][i] for i in test_seq_indices]
    targets_test = [ts_data['targets'][i] for i in test_seq_indices]
    ciks_test = df_test['cik'].tolist()
    trend_test = [ts_data['trend_up_or_down'][i] for i in test_seq_indices]
    
    print(f"\n   Train CIKs: {len(train_ciks)} companies, {len(sequences_train)} sequences")
    print(f"   Val CIKs:   {len(val_ciks)} companies, {len(sequences_val)} sequences")
    print(f"   Test CIKs:  {len(test_ciks)} companies, {len(sequences_test)} sequences")
    
    input_size = len(ts_data['feature_names'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train and evaluate LSTM
    print("\n" + "="*70)
    print("METHOD 1: LSTM Sequence Model")
    print("="*70)
    
    print(f"   🔨 Training LSTM model...")
    lstm_model = train_lstm_model(
        sequences_train, targets_train, sequences_val, targets_val,
        input_size=input_size, hidden_size=64, num_layers=3,
        epochs=300, batch_size=32, device=device,
        patience=20, min_delta=50.0  # Early stopping: wait 20 epochs, require 50 improvement
    )
    
    print("   ✅ LSTM training completed")
    
    # Evaluate on test sequences
    print(f"\n   📊 Evaluating LSTM on test data...")
    test_sequences_scaled = [ts_data['scaler'].transform(seq) for seq in sequences_test]
    
    y_pred, y_true = evaluate_sequence_model(
        lstm_model, test_sequences_scaled, targets_test, device, "LSTM"
    )
    
    # Create DataFrame with trend_up_or_down for ROC-AUC calculation
    df_test_for_eval = pd.DataFrame({
        'trend_up_or_down': trend_test[:len(y_pred)]  # Align length with predictions
    })
    evaluate_predictions(y_true, y_pred, df_test=df_test_for_eval)
    
    # Train and evaluate GRU
    print("\n" + "="*70)
    print("METHOD 2: GRU Sequence Model")
    print("="*70)
    
    print(f"   🔨 Training GRU model...")
    gru_model = train_gru_model(
        sequences_train, targets_train, sequences_val, targets_val,
        input_size=input_size, hidden_size=64, num_layers=3,
        epochs=300, batch_size=32, device=device,
        patience=20, min_delta=50.0  # Early stopping: wait 20 epochs, require 50 improvement
    )
    
    print("   ✅ GRU training completed")
    
    # Evaluate on test sequences
    print(f"\n   📊 Evaluating GRU on test data...")
    test_sequences_scaled = [ts_data['scaler'].transform(seq) for seq in sequences_test]
    
    y_pred, y_true = evaluate_sequence_model(
        gru_model, test_sequences_scaled, targets_test, device, "GRU"
    )
    
    # Create DataFrame with trend_up_or_down for ROC-AUC calculation
    df_test_for_eval = pd.DataFrame({
        'trend_up_or_down': trend_test[:len(y_pred)]  # Align length with predictions
    })
    evaluate_predictions(y_true, y_pred, df_test=df_test_for_eval)
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

