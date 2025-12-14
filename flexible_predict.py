#!/usr/bin/env python
"""Flexible prediction script that handles variable column counts.

This script can handle datasets with more or less than 41 columns by:
- Auto-detecting column count
- Padding or truncating to match model requirements
- Providing detailed feedback about transformations
"""

import argparse
import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def analyze_dataset(input_path: str):
    """Analyze the dataset to determine its structure."""
    try:
        # Try reading first few rows to understand structure
        df_sample = pd.read_csv(input_path, header=None, nrows=5, low_memory=False)
        num_cols = df_sample.shape[1]
        
        # Count total rows
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            num_rows = sum(1 for _ in f)
        
        # Check if first row is header - look for common header patterns
        has_header = False
        try:
            first_row = df_sample.iloc[0]
            # Check if first row contains mostly text/alpha values (likely header)
            text_count = 0
            for val in first_row:
                val_str = str(val).strip()
                # Check if value is mostly alphabetic (allowing spaces, underscores, hyphens)
                cleaned = val_str.replace(' ', '').replace('_', '').replace('-', '').lower()
                if cleaned and (cleaned.isalpha() or any(kw in cleaned for kw in ['port', 'protocol', 'service', 'source', 'destination', 'duration', 'label', 'feature', 'flow'])):
                    text_count += 1
            
            # Also check for common header keywords in first value
            first_val = str(first_row.iloc[0]).strip().lower() if len(first_row) > 0 else ""
            header_keywords = ['duration', 'port', 'protocol', 'service', 'source', 'destination', 'label', 'feature', 'flow']
            
            # More lenient detection: if first value contains header keywords OR many columns are text
            if any(keyword in first_val for keyword in header_keywords) or text_count > num_cols * 0.2:
                has_header = True
                num_rows -= 1
                logging.info(f"Detected header row (text_count={text_count}/{num_cols}): {first_row.iloc[0] if len(first_row) > 0 else 'N/A'}")
        except Exception as e:
            logging.warning(f"Could not detect header: {e}")
        
        return {
            'num_columns': num_cols,
            'estimated_rows': num_rows,
            'has_header': has_header,
            'sample_columns': num_cols
        }
    except Exception as e:
        logging.error(f"Error analyzing dataset: {e}")
        return None

def prepare_features(df: pd.DataFrame, target_cols: int = 41, categorical_indices: list = [1, 2, 3]):
    """Prepare features by padding or truncating to target column count.
    
    Args:
        df: Input dataframe
        target_cols: Target number of columns (default 41 for NSL-KDD)
        categorical_indices: Indices of categorical columns (for validation)
    
    Returns:
        Prepared dataframe with exactly target_cols columns, all numeric
    """
    current_cols = df.shape[1]
    
    # Ensure all columns are numeric before processing
    for col_idx in range(current_cols):
        if df.iloc[:, col_idx].dtype == 'object':
            # Try to convert to numeric
            df.iloc[:, col_idx] = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')
            df.iloc[:, col_idx] = df.iloc[:, col_idx].fillna(0)
    
    if current_cols == target_cols:
        logging.info(f"Dataset has exactly {target_cols} columns - no transformation needed")
        return df.astype(float)
    
    if current_cols < target_cols:
        # Pad with zeros
        padding_needed = target_cols - current_cols
        logging.info(f"Dataset has {current_cols} columns, padding with {padding_needed} zero columns")
        padding = pd.DataFrame(np.zeros((df.shape[0], padding_needed)), index=df.index, dtype=float)
        result = pd.concat([df, padding], axis=1)
        return result.astype(float)
    else:
        # Truncate to target_cols
        logging.info(f"Dataset has {current_cols} columns, truncating to {target_cols} columns")
        result = df.iloc[:, :target_cols]
        return result.astype(float)

def predict_flexible(model_path: str, label_map_path: str, input_path: str, 
                     output_path: str = None, num_features: int = None):
    """Predict with flexible column handling.
    
    Args:
        model_path: Path to trained model
        label_map_path: Path to label map
        input_path: Path to input dataset
        output_path: Path to save predictions
        num_features: Number of features to use (auto-detect if None)
    """
    # Load model
    logging.info(f"Loading model from {model_path}")
    pipeline = load(model_path)
    label_info = load(label_map_path)
    classes = label_info["classes_"]
    
    # Analyze dataset
    analysis = analyze_dataset(input_path)
    if not analysis:
        raise ValueError("Failed to analyze dataset")
    
    logging.info(f"Dataset analysis: {analysis['num_columns']} columns, ~{analysis['estimated_rows']} rows")
    
    # Read dataset with better type handling
    df = pd.read_csv(input_path, header=None, low_memory=False)
    
    # Remove header if present - do this FIRST before any processing
    if analysis['has_header']:
        df = df.iloc[1:, :].reset_index(drop=True)
        logging.info("Removed header row")
    
    # Double-check: if first row still looks like a header, remove it
    try:
        if df.shape[0] > 0 and df.shape[1] > 0:
            first_val = str(df.iloc[0, 0]).strip().lower()
            header_keywords = ['duration', 'port', 'protocol', 'service', 'source', 'destination', 'label', 'feature', 'flow']
            if any(keyword in first_val for keyword in header_keywords):
                logging.warning("Found header-like value in first data row, removing it")
                df = df.iloc[1:, :].reset_index(drop=True)
    except Exception as e:
        logging.debug(f"Header double-check: {e}")
    
    # Convert columns to numeric where possible (handle mixed types)
    # Note: The model's preprocessor will handle categorical encoding
    # For now, we'll convert everything to numeric and let the preprocessor handle it
    logging.info("Converting columns to numeric where possible...")
    
    for col_idx in range(min(df.shape[1], num_features)):
        try:
            # Convert to numeric, coercing errors to NaN
            df.iloc[:, col_idx] = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')
        except Exception as e:
            logging.warning(f"Could not convert column {col_idx} to numeric: {e}")
            df.iloc[:, col_idx] = 0
    
    # Replace infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    logging.info(f"DataFrame shape after cleaning: {df.shape}")
    
    # Determine number of features to use
    if num_features is None:
        # Try to detect from model
        try:
            # Get the preprocessor to see expected input
            preprocessor = pipeline.named_steps.get('preprocess')
            if preprocessor:
                # Try to infer from transformer
                num_features = 41  # Default for NSL-KDD
            else:
                num_features = 41
        except:
            num_features = 41
    
    logging.info(f"Using {num_features} features for prediction")
    
    # Prepare features - keep categorical columns (1,2,3) as strings, convert others to numeric
    # The model expects columns 1, 2, 3 to be categorical (strings) for OneHotEncoder
    X = df.iloc[:, :num_features].copy()
    
    categorical_indices = [1, 2, 3]  # These must be strings for OneHotEncoder
    
    logging.info("Preparing features: keeping categorical columns as strings, converting others to numeric...")
    
    for col_idx in range(X.shape[1]):
        if col_idx in categorical_indices:
            # Keep categorical columns as strings - convert to string if needed
            # If they're already numeric, convert to string (e.g., 1 -> "1")
            X.iloc[:, col_idx] = X.iloc[:, col_idx].astype(str)
            # Replace 'nan' strings with a default value
            X.iloc[:, col_idx] = X.iloc[:, col_idx].replace(['nan', 'NaN', 'None', ''], 'other')
            logging.info(f"Column {col_idx} kept as categorical (string)")
        else:
            # Convert numeric columns
            try:
                X.iloc[:, col_idx] = pd.to_numeric(X.iloc[:, col_idx], errors='coerce')
            except Exception as e:
                logging.warning(f"Could not convert column {col_idx} to numeric: {e}")
                X.iloc[:, col_idx] = 0
    
    # Handle infinity and NaN for numeric columns only
    numeric_cols = [i for i in range(X.shape[1]) if i not in categorical_indices]
    for col_idx in numeric_cols:
        # Replace infinity
        X.iloc[:, col_idx] = X.iloc[:, col_idx].replace([np.inf, -np.inf], np.nan)
        # Clip extremely large values
        max_float = np.finfo(np.float64).max * 0.9
        X.iloc[:, col_idx] = X.iloc[:, col_idx].clip(lower=-max_float, upper=max_float)
        # Fill NaN with 0
        X.iloc[:, col_idx] = X.iloc[:, col_idx].fillna(0)
        # Convert to float64
        X.iloc[:, col_idx] = X.iloc[:, col_idx].astype(np.float64)
    
    # If we have fewer columns, pad with zeros
    if X.shape[1] < num_features:
        padding_needed = num_features - X.shape[1]
        padding = pd.DataFrame(np.zeros((X.shape[0], padding_needed)), dtype=np.float64)
        X = pd.concat([X, padding], axis=1)
    
    # Final validation
    for col_idx in range(X.shape[1]):
        if col_idx in categorical_indices:
            # Ensure categorical columns are strings
            X.iloc[:, col_idx] = X.iloc[:, col_idx].astype(str)
        else:
            # Ensure numeric columns are float64
            if X.iloc[:, col_idx].dtype != np.float64:
                X.iloc[:, col_idx] = pd.to_numeric(X.iloc[:, col_idx], errors='coerce').fillna(0).astype(np.float64)
    
    logging.info(f"Running predictions on {X.shape[0]} samples with {X.shape[1]} features...")
    logging.info(f"Data types: {X.dtypes.value_counts().to_dict()}")
    
    # Log categorical column info
    cat_cols = [i for i in [1, 2, 3] if i < X.shape[1]]
    if cat_cols:
        logging.info(f"Categorical columns (as strings): {cat_cols}")
        for col_idx in cat_cols:
            unique_vals = X.iloc[:, col_idx].unique()[:10]  # First 10 unique values
            logging.info(f"  Column {col_idx} unique values (sample): {unique_vals}")
    
    # Log numeric column info
    numeric_cols = [i for i in range(X.shape[1]) if i not in [1, 2, 3]]
    if numeric_cols:
        numeric_data = X.iloc[:, numeric_cols]
        logging.info(f"Numeric columns: {numeric_cols[:10]}... (showing first 10)")
        logging.info(f"Min values: {numeric_data.min().min()}, Max values: {numeric_data.max().max()}")
        logging.info(f"Has NaN: {numeric_data.isnull().any().any()}, Has Inf: {np.isinf(numeric_data.values).any() if len(numeric_cols) > 0 else False}")
    
    # Check if we need batch processing (for HDC model with large datasets)
    # HDC model with dim=10000 needs ~12.9 GB for 692k samples, so we'll batch process
    # Use smaller batches for very large datasets to reduce processing time per batch
    n_samples = X.shape[0]
    if n_samples > 200000:
        batch_size = 25000  # Smaller batches for very large datasets
    elif n_samples > 100000:
        batch_size = 50000  # Medium batches
    else:
        batch_size = 100000  # Larger batches for smaller datasets
    
    # Predict in batches if dataset is large
    try:
        if n_samples > batch_size and hasattr(pipeline.named_steps["model"], "predict_proba"):
            logging.info(f"Large dataset detected ({n_samples} samples). Processing in batches of {batch_size}...")
            all_probs = []
            all_preds = []
            all_max_prob = []
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_X = X.iloc[i:end_idx, :]
                logging.info(f"Processing batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size} (rows {i} to {end_idx-1})...")
                
                batch_probs = pipeline.predict_proba(batch_X)
                batch_preds = np.argmax(batch_probs, axis=1)
                batch_max_prob = batch_probs.max(axis=1)
                
                all_probs.append(batch_probs)
                all_preds.append(batch_preds)
                all_max_prob.append(batch_max_prob)
            
            # Concatenate results
            probs = np.vstack(all_probs)
            preds = np.concatenate(all_preds)
            max_prob = np.concatenate(all_max_prob)
            logging.info("Batch processing complete!")
        else:
            # Process all at once for smaller datasets
            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                probs = pipeline.predict_proba(X)
                preds = np.argmax(probs, axis=1)
                max_prob = probs.max(axis=1)
            else:
                preds = pipeline.predict(X)
                max_prob = np.ones(len(preds))
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        logging.error(f"X shape: {X.shape}")
        logging.error(f"X dtypes:\n{X.dtypes}")
        logging.error(f"X min/max:\nMin: {X.min()}\nMax: {X.max()}")
        logging.error(f"Sample of X (first 5 rows, first 10 cols):\n{X.iloc[:5, :10]}")
        raise
    
    pred_labels = pd.Series(classes[preds], name="predicted_label")
    conf = pd.Series(max_prob, name="confidence")
    
    # Combine results
    result = pd.concat([df.reset_index(drop=True), pred_labels, conf], axis=1)
    
    if output_path:
        result.to_csv(output_path, index=False, header=False)
        logging.info(f"Wrote {len(result)} predictions to {output_path}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Flexible prediction with variable columns")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--label_map_path", required=True, help="Path to label map")
    parser.add_argument("--input_path", required=True, help="Path to input dataset")
    parser.add_argument("--output_path", required=True, help="Path to save predictions")
    parser.add_argument("--num_features", type=int, default=None, 
                        help="Number of features to use (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    try:
        result = predict_flexible(
            args.model_path,
            args.label_map_path,
            args.input_path,
            args.output_path,
            args.num_features
        )
        logging.info(f"Successfully processed {len(result)} rows")
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

