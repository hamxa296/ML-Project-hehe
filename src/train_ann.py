import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Paths
DATA_DIR = os.path.abspath('data')
TRAIN_CSV = os.path.join(DATA_DIR, 'train_balanced.csv')
TEST_CSV  = os.path.join(DATA_DIR, 'test.csv')

def train_ann():
    print("--- Starting ANN Training Script ---")
    print("Loading datasets...")
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    X_train = train.drop(columns=['isFraud'])
    y_train = train['isFraud']
    X_test  = test.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
    y_test  = test['isFraud']

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Building model...")
    model = models.Sequential([
        layers.Dense(256, input_dim=X_train.shape[1], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', curve='PR')]
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_auc', 
        patience=5, 
        restore_best_weights=True,
        mode='max'
    )

    print("Starting fit (50 epochs)...")
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=2048,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    print("\n--- Final Evaluation ---")
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print("--- Training Complete ---")

if __name__ == "__main__":
    train_ann()
