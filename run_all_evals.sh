#!/bin/bash

# List of notebooks to execute (EXCLUDING 05_Neural_Network)
NOTEBOOKS=(
    "notebooks/04_Logistic_Regression.ipynb"
    "notebooks/06_SVM.ipynb"
    "notebooks/07_Random_Forest.ipynb"
    "notebooks/08_XGBoost.ipynb"
    "notebooks/09_KNN.ipynb"
    "notebooks/10_Advanced_Experiments.ipynb"
    "notebooks/11_Final_Champion_Duel.ipynb"
)

for NB in "${NOTEBOOKS[@]}"; do
    if [ -f "$NB" ]; then
        echo "Executing $NB..."
        # Use --timeout -1 to prevent long-running models from being killed
        jupyter nbconvert --to notebook --execute "$NB" --inplace --ExecutePreprocessor.timeout=-1
    else
        echo "Warning: $NB not found."
    fi
done

echo "Batch execution complete."
