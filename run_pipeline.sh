#!/bin/bash
set -e

echo "1. Starting Preprocessing..."
/usr/local/bin/python3 -m jupyter nbconvert --to notebook --execute notebooks/01_Preprocessing.ipynb --inplace

echo "2. Starting Post-Preprocessing EDA..."
/usr/local/bin/python3 -m jupyter nbconvert --to notebook --execute notebooks/02_EDA.ipynb --inplace

echo "3. Starting Data Splitting & Balancing..."
/usr/local/bin/python3 -m jupyter nbconvert --to notebook --execute notebooks/03_Data_Splitting_and_Balancing.ipynb --inplace

echo "✅ All notebooks executed successfully."
