import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Style configuration
plt.style.use('ggplot')
sns.set_palette('viridis')
FIG_DIR = 'reports/figures/raw_eda/'
os.makedirs(FIG_DIR, exist_ok=True)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def save_plot(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

def main():
    data_dir = 'data/'
    print("Loading raw data...")
    train_trans = pd.read_csv(os.path.join(data_dir, 'train_transaction.csv'))
    train_id = pd.read_csv(os.path.join(data_dir, 'train_identity.csv'))
    
    print("Optimizing memory...")
    train_trans = reduce_mem_usage(train_trans)
    train_id = reduce_mem_usage(train_id)
    
    print("Merging data...")
    df = train_trans.merge(train_id, on='TransactionID', how='left')
    
    # 1. Class Imbalance Plot
    print("Plotting Class Imbalance...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='isFraud', data=df)
    plt.title('Target Distribution (isFraud)')
    plt.yscale('log')
    save_plot('01_class_imbalance.png')
    
    # 2. Transaction Amount Distribution
    print("Plotting Transaction Amount Distribution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(df['TransactionAmt'], bins=50, ax=ax1)
    ax1.set_title('Transaction Amount Distribution')
    sns.histplot(np.log1p(df['TransactionAmt']), bins=50, ax=ax2)
    ax2.set_title('Transaction Amount (Log) Distribution')
    save_plot('02_transaction_amt.png')
    
    # 3. Fraud over Time
    print("Plotting Fraud over Time...")
    plt.figure(figsize=(15, 6))
    df['TransactionDT'].hist(bins=100, alpha=0.5, label='Total')
    df[df['isFraud']==1]['TransactionDT'].hist(bins=100, alpha=0.8, label='Fraud')
    plt.title('Transactions over Time (TransactionDT)')
    plt.legend()
    save_plot('03_fraud_over_time.png')
    
    # 4. Missingness vs Fraud
    print("Plotting Missingness vs Fraud...")
    # Select columns with high nulls to see if missingness correlates with fraud
    null_cols = df.isnull().mean().sort_values(ascending=False).head(10).index
    missing_df = df[null_cols].isnull().astype(int)
    missing_df['isFraud'] = df['isFraud']
    corr_missing = missing_df.corr()['isFraud'].drop('isFraud')
    plt.figure(figsize=(10, 6))
    corr_missing.plot(kind='barh')
    plt.title('Correlation of Missingness with Fraud (Top 10 Null Columns)')
    save_plot('04_missingness_vs_fraud.png')
    
    # 5. Categorical Fraud Rates
    print("Plotting Categorical Fraud Rates...")
    cat_cols = ['ProductCD', 'card4', 'card6', 'DeviceType']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for i, col in enumerate(cat_cols):
        if col in df.columns:
            sns.barplot(x=col, y='isFraud', data=df, ax=axes[i//2, i%2])
            axes[i//2, i%2].set_title(f'Fraud Rate by {col}')
    save_plot('05_categorical_fraud_rates.png')
    
    # 6. PCA Visualization (Bonus)
    print("Plotting PCA (V-features)...")
    v_cols = [c for c in df.columns if c.startswith('V')][:50] # Use subset of V-features
    pca_data = df[v_cols].fillna(df[v_cols].median())
    pca_data = StandardScaler().fit_transform(pca_data)
    pca = PCA(n_components=2)
    components = pca.fit_transform(pca_data)
    plt.figure(figsize=(10, 8))
    plt.scatter(components[:, 0], components[:, 1], c=df['isFraud'], alpha=0.1, cmap='RdBu')
    plt.title('PCA of V-Features (First 50)')
    plt.colorbar(label='isFraud')
    save_plot('06_pca_visualization.png')
    
    # 7. Correlation Heatmap (Bonus)
    print("Plotting Correlation Heatmap...")
    cols_for_corr = ['TransactionAmt', 'dist1', 'dist2', 'C1', 'C2', 'D1', 'D2', 'isFraud']
    corr_matrix = df[cols_for_corr].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Selected Features')
    save_plot('07_correlation_heatmap.png')
    
    # 8. Feature Importance (Bonus)
    print("Plotting Feature Importance...")
    # Sample data for speed
    sample_df = df.sample(50000, random_state=42).fillna(0)
    X_temp = sample_df.select_dtypes(exclude=['object']).drop('isFraud', axis=1)
    y_temp = sample_df['isFraud']
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_temp, y_temp)
    importances = pd.Series(rf.feature_importances_, index=X_temp.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 8))
    importances.plot(kind='barh')
    plt.title('Top 20 Feature Importance (Random Forest)')
    save_plot('08_feature_importance.png')
    
    # 9. Identity Feature Distributions (Bonus)
    print("Plotting Identity Features...")
    if 'DeviceType' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(x='DeviceType', hue='isFraud', data=df)
        plt.title('Fraud Distribution by DeviceType')
        plt.yscale('log')
        save_plot('09_identity_distribution.png')

    print("\nEDA script completed. All figures are in reports/figures/raw_eda/")

if __name__ == '__main__':
    main()
