"""
Create year-based splits from processed training data.

This script creates training splits based on years to simulate realistic data accumulation over time.

Test data is always the same to ensure consistency for model comparison.

Usage:
    python src/data/training_data_splits_by_year.py
"""

import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import PARAMS

# Get parameters
TEST_SIZE = PARAMS['data']['test_size']
RANDOM_STATE = PARAMS['data']['random_state']

# Split function
def training_data_splits_by_year():

    # ==================== Step 1 ====================
    # Load processed data
    try:
        X_train_full = pd.read_csv('data/processed/X_train.csv')
        y_train_full = pd.read_csv('data/processed/y_train.csv')
        X_test_fixed = pd.read_csv('data/processed/X_test.csv')
        y_test_fixed = pd.read_csv('data/processed/y_test.csv')
        
        # Check if Year column exists
        if 'Year' not in X_train_full.columns:
            print("ERROR: Year column not found in X_train!")
            print("Ensure preprocessing keeps Year column.")
            return False
            
    except FileNotFoundError as e:
        print("ERROR: Processed data not found!")
        print("Run preprocessing first:")
        print("python src/data/preprocess.py")
        print(f"\nError: {e}")
        return False
    
    print(f"Successfully loaded:")
    print(f"X_train: {X_train_full.shape}")
    print(f"y_train: {y_train_full.shape}")
    print(f"X_test:  {X_test_fixed.shape}")
    print(f"y_test:  {y_test_fixed.shape}")


    # Check if Year is scaled or not    
    year_min = X_train_full['Year'].min()
    year_max = X_train_full['Year'].max()
    
    if year_max < 100:
        print("ERROR: Year column appears to be scaled.")
        print(f"Year range: {year_min:.2f} to {year_max:.2f}")
        print("Modify preprocess.py to NOT scale Year:")
        print("Remove 'Year' from numerical_cols_scale list")
        print("Then re-run: python src/data/preprocess.py")
        return False
    
    print("Year column is not scaled")
    print(f"Year range: {int(year_min)} to {int(year_max)}")


    # ==================== Step 2 ====================
    # Analyze and filter years
    X_train_full['Year'] = X_train_full['Year'].round().astype(int)
    
    # Show distribution
    year_counts = X_train_full['Year'].value_counts().sort_index()
    print("Samples per year in training data:")
    for year, count in year_counts.items():
        print(f"{year}: {count:6d} samples")
    
    # Use only complete years (2008-2016)
    print("Removing incomplete years.")
    
    years_before = len(X_train_full)
    mask = (X_train_full['Year'] >= 2008) & (X_train_full['Year'] <= 2016)
    X_train_full = X_train_full[mask].reset_index(drop=True)
    y_train_full = y_train_full[mask].reset_index(drop=True)    
    years_available = sorted(X_train_full['Year'].unique())
    print(f"Filtered years: {years_available}")
    print(f"Removed {years_before - len(X_train_full)} samples")
    print(f"Remaining: {len(X_train_full)} samples")


    # ==================== Step 3 ====================
    # Calculate class distribution in 100% training data
    class_dist = y_train_full['RainTomorrow'].value_counts(normalize=True).sort_index()
    print("Overall class distribution: ", class_dist)


    # ==================== Step 4 ====================
    # Create output directory
    splits_dir = Path('data/training_data_splits_by_year')
    splits_dir.mkdir(exist_ok=True, parents=True)


    # ==================== Step 5 ====================
    # Initialize metadata
    metadata = {
        'total_train_samples': len(X_train_full),
        'test_samples': len(X_test_fixed),
        'random_state': RANDOM_STATE,
        'split_method': 'cumulative_by_year',
        'years_available': [int(y) for y in years_available],
        'n_features': len(X_train_full.columns) - 1,  # Exclude Year
        'splits': []
    }


    # ==================== Step 6 ====================
    # Create year-based splits
    for i, end_year in enumerate(years_available, 1):
        
        # Select all data from first year up to last year    
        start_year = years_available[0]
        mask = (X_train_full['Year'] >= start_year) & (X_train_full['Year'] <= end_year)
        
        X_train_split = X_train_full[mask].reset_index(drop=True)
        y_train_split = y_train_full[mask].reset_index(drop=True)
        
        # Create split directory name
        if start_year == end_year:
            split_name = f"{end_year}"
        else:
            split_name = f"{start_year}-{end_year}"
        
        split_dir = splits_dir / f'split_{i:02d}_{split_name}'
        split_dir.mkdir(exist_ok=True)
        
        # Save training data (grows year by year)
        X_train_split.to_csv(split_dir / 'X_train.csv', index=False)
        y_train_split.to_csv(split_dir / 'y_train.csv', index=False)
        
        # Save test data (ALWAYS the same!)
        X_test_fixed.to_csv(split_dir / 'X_test.csv', index=False)
        y_test_fixed.to_csv(split_dir / 'y_test.csv', index=False)
        
        # Calculate class distribution for this split
        n_no_rain = (y_train_split['RainTomorrow'] == 0).sum()
        n_rain = (y_train_split['RainTomorrow'] == 1).sum()
        pct_no_rain = n_no_rain / len(y_train_split) * 100
        pct_rain = n_rain / len(y_train_split) * 100
        
        # Store metadata
        metadata['splits'].append({
            'split_id': i,
            'years': split_name,
            'start_year': int(start_year),
            'end_year': int(end_year),
            'train_samples': len(X_train_split),
            'test_samples': len(X_test_fixed),
            'class_distribution': {
                'no_rain': float(pct_no_rain / 100),
                'rain': float(pct_rain / 100)
            }
        })
        
        # Display progress
        print(f"   Split {i:02d} ({split_name:11s}): "
              f"{len(X_train_split):6d} train, {len(X_test_fixed):6d} test │ "
              f"No Rain: {pct_no_rain:.1f}%, Rain: {pct_rain:.1f}%")


    # ==================== Step 7 ====================
    # Save metadata
    metadata_path = splits_dir / 'metadata.yaml'
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    print(f"Metadata saved: {metadata_path}")


    return True

if __name__ == "__main__":

    success = training_data_splits_by_year()

    if success:
        print("SUCCESS!")

    else:
        print("FAILED - check errors above")
        sys.exit(1)
