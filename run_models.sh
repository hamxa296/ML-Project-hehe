#!/bin/bash
echo "Starting Logistic Regression..."
/usr/local/bin/python3 -m jupyter nbconvert --to notebook --execute notebooks/04_Logistic_Regression.ipynb --inplace &

echo "Starting Random Forest..."
/usr/local/bin/python3 -m jupyter nbconvert --to notebook --execute notebooks/07_Random_Forest.ipynb --inplace &

echo "Starting XGBoost..."
/usr/local/bin/python3 -m jupyter nbconvert --to notebook --execute notebooks/08_XGBoost.ipynb --inplace &

wait
echo "Baseline models finished."

echo "Starting SVM (with PCA)..."
/usr/local/bin/python3 -m jupyter nbconvert --to notebook --execute notebooks/06_SVM.ipynb --inplace

echo "✅ All model notebooks executed."
