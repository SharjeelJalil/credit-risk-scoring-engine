# -*- coding: utf-8 -*-
"""
Limit Determination Engine
============================
Converts ML credit risk scorings into actionable credit decisions by:
  1. Estimating proxy income from 6-month average deposit balance
  2. Mapping customers to salary bands
  3. Looking up credit multipliers from the risk x salary band matrix
  4. Computing final credit limits

The multiplier matrix is defined by the Digital Lending team and reflects
the bank's risk appetite: low-risk customers in high salary bands get
higher multiples (up to 5x income), while very high-risk customers in
low salary bands get zero.

@author: sharjeel.jalil
"""

import numpy as np
import pandas as pd
import sqlalchemy


# ── Proxy Income Query ──────────────────────────────────────────────────
PROXY_INCOME_QUERY = """
SELECT [ACCOUNT_NUM], avg([Avg_Bal]) as est_salary
FROM [ADV_CC].[dbo].[customer_deposits] cd
WHERE 1=1
  AND [Month_deposit] >= DateAdd(month, -6,
      Convert(date, datefromparts(YEAR(GETDATE()), MONTH(GETDATE()), 1)))
  AND [Month_deposit] < datefromparts(YEAR(GETDATE()), MONTH(GETDATE()), 1)
  AND (len(id_number) = 13 AND patindex('[^0-9]', id_number) = 0)
GROUP BY ACCOUNT_NUM
"""


def get_proxy_income(conn_string):
    """
    Estimate monthly income from 6-month average deposit balance.

    Uses the bank's deposit system to compute the average month-end
    balance over the last 6 months as a proxy for monthly salary.
    """
    conn = sqlalchemy.create_engine(conn_string)
    avg_bal_data = pd.read_sql(PROXY_INCOME_QUERY, conn)
    return avg_bal_data


# ── Risk x Salary Band Multiplier Matrix ────────────────────────────────
MULTIPLIER_MATRIX = pd.DataFrame({
    'RiskBand': [
        'Low-Medium', 'High', 'Very High',
        'Low-Medium', 'High', 'Very High',
        'Low-Medium', 'High', 'Very High',
        'Low-Medium', 'High', 'Very High',
        'Low-Medium', 'High', 'Very High',
        'Low-Medium', 'High', 'Very High',
        'Low-Medium', 'High', 'Very High',
        'Low-Medium', 'High', 'Very High',
        'Low-Medium', 'High', 'Very High',
    ],
    'SalaryBand': [
        '>20>30', '>20>30', '>20>30',
        '>30>40', '>30>40', '>30>40',
        '>40>50', '>40>50', '>40>50',
        '>50>60', '>50>60', '>50>60',
        '>60>70', '>60>70', '>60>70',
        '>70>80', '>70>80', '>70>80',
        '>80>90', '>80>90', '>80>90',
        '>90>100', '>90>100', '>90>100',
        '>100', '>100', '>100',
    ],
    'Multiplier': [
        1.5, 0.5, 0,
        1.5, 0.5, 0,
        2, 1, 0.5,
        2, 1, 0.5,
        2.5, 1, 0.5,
        3, 1, 0.5,
        4, 1, 0.5,
        4, 1, 0.5,
        5, 1, 0.5,
    ]
})

# Salary band edges (PKR thousands)
SALARY_EDGES = [-1, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, np.inf]
SALARY_LABELS = ['<=20', '>20>30', '>30>40', '>40>50', '>50>60',
                 '>60>70', '>70>80', '>80>90', '>90>100', '>100']

# Risk band edges (default probability)
RISK_EDGES = [0, 0.6, 0.8, 1]
RISK_LABELS = ['Low-Medium', 'High', 'Very High']


def determine_limits(scored_df, proxy_income_df):
    """
    Compute credit limits for all scored customers.

    Steps:
      1. Merge scored results with proxy income data
      2. Assign salary bands based on estimated income
      3. Assign risk bands based on default probability
      4. Look up multiplier from the risk x salary matrix
      5. Compute limit = estimated_salary x multiplier

    Returns DataFrame with all scoring and limit columns.
    """
    # Merge with proxy income
    main_df = pd.merge(
        scored_df,
        proxy_income_df,
        on='ACCOUNT_NUM',
        how='left'
    )

    # Assign salary bands
    main_df['SalaryBand'] = pd.cut(
        main_df['est_salary'],
        bins=SALARY_EDGES,
        labels=SALARY_LABELS
    )

    # Assign risk bands (collapsed from 5-tier to 3-tier for limit lookup)
    main_df['RiskBand'] = pd.cut(
        main_df['DefaultProb'],
        bins=RISK_EDGES,
        labels=RISK_LABELS
    )

    # Look up multiplier
    main_df = pd.merge(
        main_df,
        MULTIPLIER_MATRIX,
        how='left',
        on=['SalaryBand', 'RiskBand']
    )

    # Compute final limit
    main_df['Limit'] = main_df['est_salary'] * main_df['Multiplier']

    return main_df


def print_decision_summary(limit_df):
    """Print accept/reject summary."""
    accepted = limit_df[limit_df['Limit'] > 0]
    rejected = limit_df[limit_df['Limit'].isna() | (limit_df['Limit'] == 0)]

    print("\n=== Lending Decision Summary ===")
    print(f"Accepted: {len(accepted):,}")
    print(f"Rejected: {len(rejected):,}")
    print(f"Total:    {len(limit_df):,}")
    print(f"Acceptance rate: {len(accepted)/len(limit_df)*100:.1f}%")
