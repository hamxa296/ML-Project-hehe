import pandas as pd
import os

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    print("1. Loading raw training files...")
    train_trans = pd.read_csv(os.path.join(data_dir, 'train_transaction.csv'))
    train_id = pd.read_csv(os.path.join(data_dir, 'train_identity.csv'))
    
    print("2. Loading raw testing files...")
    test_trans = pd.read_csv(os.path.join(data_dir, 'test_transaction.csv'))
    test_id = pd.read_csv(os.path.join(data_dir, 'test_identity.csv'))
    
    print("\n--- Merging Datasets ---")
    # We use a LEFT join on TransactionID because not every transaction has identity data
    train_merged = pd.merge(train_trans, train_id, on='TransactionID', how='left')
    test_merged = pd.merge(test_trans, test_id, on='TransactionID', how='left')
    
    print(f"Merged Train Shape: {train_merged.shape}")
    print(f"Merged Test Shape:  {test_merged.shape}")
    
    # Save the merged files so you can view them locally
    train_merged_path = os.path.join(data_dir, 'merged_raw_train.csv')
    test_merged_path = os.path.join(data_dir, 'merged_raw_test.csv')
    
    print(f"\nSaving merged files to data/ directory (this might take a minute, they are ~1.5GB each)...")
    train_merged.to_csv(train_merged_path, index=False)
    test_merged.to_csv(test_merged_path, index=False)
    print(f"✅ Saved: {train_merged_path}")
    print(f"✅ Saved: {test_merged_path}")
    
    print("\n========================================================")
    print("      BASIC EDA (Why Preprocessing was Needed)            ")
    print("========================================================")
    
    print("\n1. Data Types and Memory Usage")
    print("-" * 50)
    print(f"Total Memory Usage: {train_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("(This is why we implemented the 'MemoryOptimizer' downcasting step!)")
    
    print("\n2. The Missing Values Problem")
    print("-" * 50)
    null_percentages = (train_merged.isnull().sum() / len(train_merged)) * 100
    high_null_cols = null_percentages[null_percentages > 50].sort_values(ascending=False)
    print(f"Total columns: {train_merged.shape[1]}")
    print(f"Columns with MORE than 50% missing values: {len(high_null_cols)}")
    print("\nTop 10 columns with most missing values:")
    print(high_null_cols.head(10).apply(lambda x: f"{x:.2f}% null"))
    print("\n(This is why we implemented the 'DropHighNulls' step!)")
    
    print("\n3. Categorical High Cardinality")
    print("-" * 50)
    object_cols = train_merged.select_dtypes(include=['object']).columns
    print(f"Number of categorical columns: {len(object_cols)}")
    print("Unique values in some categorical columns:")
    for col in ['P_emaildomain', 'R_emaildomain', 'DeviceInfo', 'id_31']:
        if col in train_merged.columns:
            print(f" - {col}: {train_merged[col].nunique()} unique values")
    print("\n(Standard One-Hot Encoding would create hundreds of new columns here.")
    print(" This is why we implemented 'FrequencyEncoder'!)")
    
    print("\n4. Target Variable Imbalance")
    print("-" * 50)
    fraud_counts = train_merged['isFraud'].value_counts(normalize=True) * 100
    print(fraud_counts.apply(lambda x: f"{x:.2f}%").to_string())
    print("\n(This is why we implemented SMOTE + Undersampling!)")

if __name__ == '__main__':
    main()
