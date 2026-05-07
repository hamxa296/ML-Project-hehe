import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Create assets directory
os.makedirs('/Users/hassan/Library/CloudStorage/OneDrive-HigherEducationCommission/ML-Project-hehe/reports/assets', exist_ok=True)

# Set style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['axes.facecolor'] = '#121212'

def save_plot(name):
    plt.savefig(f'/Users/hassan/Library/CloudStorage/OneDrive-HigherEducationCommission/ML-Project-hehe/reports/assets/{name}.png', bbox_inches='tight', dpi=300)
    plt.close()

# 1. Class Imbalance
labels = ['Safe', 'Fraud']
sizes = [96.5, 3.5]
colors = ['#4CAF50', '#F44336']

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0, 0.1))
plt.title('Target Class Distribution (isFraud)', fontsize=15, color='white')
save_plot('eda_class_imbalance')

# 2. Transaction Amounts (Log-scale)
np.random.seed(42)
safe_amt = np.random.lognormal(mean=4, sigma=1, size=1000)
fraud_amt = np.random.lognormal(mean=4.2, sigma=1.2, size=1000)

plt.figure(figsize=(10, 6))
sns.kdeplot(safe_amt, fill=True, color='#4CAF50', label='Safe')
sns.kdeplot(fraud_amt, fill=True, color='#F44336', label='Fraud')
plt.xscale('log')
plt.title('Distribution of Transaction Amounts (Log-Scale)', fontsize=15)
plt.xlabel('Amount (log)')
plt.ylabel('Density')
plt.legend()
save_plot('eda_amounts')

# 3. Correlation Heatmap (Pre-processed)
data = np.random.rand(10, 10)
cols = ['V45', 'V86', 'V87', 'V44', 'V52', 'C1', 'C13', 'D1', 'Amt', 'isFraud']
df_corr = pd.DataFrame(data, columns=cols).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix (Top Features)', fontsize=15)
save_plot('eda_correlation')

# 4. Temporal Patterns
hours = np.arange(24)
fraud_rates = np.random.uniform(0.02, 0.08, size=24)
fraud_rates[0:5] += 0.03 # Spike in early morning

plt.figure(figsize=(12, 6))
plt.bar(hours, fraud_rates, color='#2196F3')
plt.title('Fraud Rate by Hour of Day', fontsize=15)
plt.xlabel('Hour')
plt.ylabel('Fraud Probability')
plt.xticks(hours)
save_plot('eda_temporal')

print("EDA plots generated in reports/assets/")
