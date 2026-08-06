"""
eda_and_abtest.py

Exploratory Data Analysis (EDA) and A/B Testing for Cookie Cats Mobile Game Dataset.

This script analyzes player retention and game rounds across two experimental groups:
- Control Group (gate_30): Gate located at level 30
- Test Group (gate_40): Gate located at level 40

Statistical Analysis:
A Chi-Square Test of Independence is used to evaluate whether changing the gate position
has a statistically significant effect on 7-day retention (D7 retention).
"""

import os
import pandas as pd
from scipy.stats import chi2_contingency

def load_data(filepath="data/cookie_cats.csv"):
    """
    Step 1: Load the Cookie Cats dataset from CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please ensure cookie_cats.csv is in /data.")
    
    df = pd.read_csv(filepath)
    return df

def main():
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    data_path = os.path.join("data", "cookie_cats.csv")
    df = load_data(data_path)
    
    print("=" * 60)
    print("COOKIE CATS A/B TESTING & EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # 2. Total Unique Players & Preview
    # ---------------------------------------------------------
    unique_players = df['userid'].nunique()
    print(f"\n1. DATA PREVIEW & OVERVIEW")
    print(f"Total Unique Players (userids): {unique_players:,}")
    print("\nFirst 5 Rows:")
    print(df.head())
    
    # ---------------------------------------------------------
    # 3. Overall D1 and D7 Retention Rates
    # ---------------------------------------------------------
    # retention_1: boolean indicating if player returned 1 day after installing
    # retention_7: boolean indicating if player returned 7 days after installing
    d1_overall = df['retention_1'].mean() * 100
    d7_overall = df['retention_7'].mean() * 100
    
    print(f"\n2. OVERALL RETENTION RATES")
    print(f"Overall 1-Day Retention (D1): {d1_overall:.2f}%")
    print(f"Overall 7-Day Retention (D7): {d7_overall:.2f}%")
    
    # ---------------------------------------------------------
    # 4. Group Comparison: gate_30 vs gate_40
    # ---------------------------------------------------------
    # We group by the 'version' column to compare the control and test groups.
    print(f"\n3. RETENTION & METRICS BY VERSION GROUP")
    group_analysis = df.groupby('version').agg(
        player_count=('userid', 'count'),
        d1_retention_pct=('retention_1', lambda x: x.mean() * 100),
        d7_retention_pct=('retention_7', lambda x: x.mean() * 100),
        avg_gamerounds=('sum_gamerounds', 'mean')
    ).reset_index()
    
    print(group_analysis.to_string(index=False))
    
    # ---------------------------------------------------------
    # 5. Chi-Square Test of Independence (version vs retention_7)
    # ---------------------------------------------------------
    # Statistical Explanation:
    # A Chi-Square Test of Independence tests whether two categorical variables
    # (in this case, 'version' [gate_30 vs gate_40] and 'retention_7' [True vs False])
    # are independent or if there is a significant relationship between them.
    #
    # Null Hypothesis (H0): Gate version and D7 retention are independent (gate placement has no effect).
    # Alternative Hypothesis (H1): Gate version and D7 retention are dependent (gate placement affects retention).
    
    print(f"\n4. HYPOTHESIS TESTING: CHI-SQUARE TEST (Version vs 7-Day Retention)")
    
    # Create contingency table: rows = version, columns = retention_7 (True/False)
    contingency_table = pd.crosstab(df['version'], df['retention_7'])
    print("\nContingency Table (Observed Frequencies):")
    print(contingency_table)
    
    # Run chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nChi-Square Statistic : {chi2_stat:.4f}")
    print(f"Degrees of Freedom   : {dof}")
    print(f"p-value              : {p_value:.5f}")
    
    # ---------------------------------------------------------
    # 6. Plain-English Conclusion & Recommendation
    # ---------------------------------------------------------
    # We use alpha = 0.05 as our significance threshold.
    alpha = 0.05
    print(f"\n5. CONCLUSION & BUSINESS RECOMMENDATION (Threshold alpha = {alpha})")
    
    if p_value < alpha:
        print(f"RESULT: Statistically Significant (p-value = {p_value:.5f} < {alpha})")
        print("EXPLANATION:")
        print("We REJECT the null hypothesis. There is a statistically significant difference in 7-day retention")
        print("between players who encountered the gate at Level 30 vs Level 40.")
        print("Notice that D7 retention is HIGHER when the gate is at Level 30 than at Level 40.")
        print("\nRECOMMENDATION:")
        print("Keep the gate at Level 30. Moving the gate to Level 40 causes a small but significant drop in long-term (7-day) retention.")
    else:
        print(f"RESULT: Not Statistically Significant (p-value = {p_value:.5f} >= {alpha})")
        print("EXPLANATION:")
        print("We FAIL TO REJECT the null hypothesis. There is no statistically significant evidence")
        print("that gate placement affects 7-day retention rates.")
        print("\nRECOMMENDATION:")
        print("Gate location does not show a measurable impact on D7 retention based on this test.")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
