# -*- coding: utf-8 -*-
"""
Production Scoring Pipeline
=============================
Scores the full existing customer base using the trained MLP model.
Outputs default probabilities, risk categories, and feeds into the
limit determination engine.

Pipeline:
  1. Load trained model and preprocessing artifacts (joblib)
  2. Apply same feature encoding and normalization as training
  3. Predict default probabilities for all customers
  4. Assign 5-tier risk categories based on probability thresholds
  5. Pass results to limit determination

@author: sharjeel.jalil
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn import preprocessing


def load_production_artifacts():
    """Load the serialized model, scaler, and encoder from training."""
    model = joblib.load('finalized_model.sav')
    scaler = joblib.load('MinMaxScaler.sav')
    encoder = joblib.load('LabelEncoder.sav')
    selected_columns = joblib.load('SelectedColumns.sav')
    return model, scaler, encoder, selected_columns


def prepare_scoring_data(df, selected_columns, scaler):
    """
    Apply the same preprocessing pipeline used in training:
    - Select relevant columns
    - One-hot encode categoricals
    - Fill missing values
    - Apply MinMax normalization using the TRAINING scaler
    """
    # Select features matching training
    dd = [x for x in df.columns if x not in selected_columns]
    eval_set = df.copy()
    eval_set.drop(dd, axis=1, inplace=True)

    # One-hot encode categoricals (same as training)
    xp = pd.get_dummies(eval_set[['BankingGroup', 'ProductType', 'ADDRESS_TYPE', 'gender']])
    eval_set = pd.concat([eval_set, xp], sort=False, axis=1)
    eval_set.drop(['BankingGroup', 'ProductType', 'ADDRESS_TYPE', 'gender'], axis=1, inplace=True)

    eval_set.fillna(0, inplace=True)

    # Normalize using training scaler (critical: must use same scaler)
    x_scaled = scaler.transform(eval_set.values)
    eval_set = pd.DataFrame(x_scaled, columns=eval_set.columns, index=eval_set.index)

    return eval_set


def score_customers(eval_set, model, id_data):
    """
    Generate default probabilities and risk categories.

    Risk tiers:
        Very Low:  0.0 - 0.2
        Low:       0.2 - 0.4
        Medium:    0.4 - 0.6
        High:      0.6 - 0.8
        Very High: 0.8 - 1.0
    """
    probabilities = model.predict_proba(eval_set.values)[:, 1]
    predictions = model.predict(eval_set.values)

    result = id_data.copy()
    result['DefaultProb'] = probabilities
    result['PredictedStatus'] = predictions

    result['Risk Category'] = pd.cut(
        result['DefaultProb'],
        bins=[-1, 0.2, 0.4, 0.6, 0.8, np.inf],
        labels=['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    )

    return result


def print_risk_summary(result_set):
    """Print the risk distribution summary."""
    summary = result_set.groupby('Risk Category').agg(
        Total_Customers=('DefaultProb', 'count'),
        Pct_of_Total=('DefaultProb', lambda x: f"{len(x)/len(result_set)*100:.2f}%"),
        Avg_Default_Prob=('DefaultProb', 'mean')
    )
    print("\n=== Risk Distribution ===")
    print(summary)
    print(f"\nTotal customers scored: {len(result_set):,}")
