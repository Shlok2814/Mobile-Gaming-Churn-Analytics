# 🎮 Mobile Gaming Player Churn & Retention Analytics

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-3F4F75.svg)

LINK- https://mobile-gaming-churn-analytics.streamlit.app/
---

## 📌 Project Overview & Problem Statement

In the mobile gaming industry, player retention is the single most critical driver of lifetime value (LTV) and monetization. Puzzle games like *Cookie Cats* rely on level gates—mechanics that force players to wait or make in-app purchases to continue playing.

This project addresses two core business challenges:
1. **A/B Test Analysis**: Evaluating whether moving the first progression gate from **Level 30** to **Level 40** increases or decreases player retention rates at Day 1 (D1) and Day 7 (D7).
2. **Predictive Churn Modeling**: Building a Machine Learning classifier to identify high-risk players likely to churn within 7 days based on early week-1 engagement signals, enabling proactive retention offers.

---

## 📊 Dataset Description

The dataset comes from Kaggle's [Cookie Cats Mobile Games A/B Testing](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing) experiment.

- **Total Players**: 90,189 unique installations
- **Features (5 columns)**:
  - `userid`: Unique identifier for each player.
  - `version`: Experimental group assignment (`gate_30` = Control at Level 30, `gate_40` = Test at Level 40).
  - `sum_gamerounds`: Total game rounds played during the first week after installation.
  - `retention_1`: Boolean (`True`/`False`), whether the player returned 1 day after installing.
  - `retention_7`: Boolean (`True`/`False`), whether the player returned 7 days after installing.

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9 or higher installed.

### 2. Virtual Environment Setup
```powershell
# Clone or navigate to the project root
cd mobile-gaming-churn-analytics

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🏃 Executing the Project Pipeline

Run the scripts in the following order to execute the analysis, train the model, and launch the interactive web dashboard:

### Step 1: Validate Dataset
```powershell
python check_data.py
```
*Validates dataset structure, column data types, missing values, and checks for duplicate user IDs.*

### Step 2: Run Exploratory Data Analysis & Hypothesis Testing
```powershell
python eda_and_abtest.py
```
*Performs summary statistics, calculates D1 & D7 retention rates, executes a Chi-Square Test of Independence ($p < 0.05$), and prints statistical conclusions.*

### Step 3: Train the Machine Learning Churn Model
```powershell
python churn_model.py
```
*Engineers churn features, removes outlier bots (`sum_gamerounds >= 5000`), trains a `RandomForestClassifier`, prints evaluation metrics (Accuracy, ROC-AUC, Classification Report), and saves `/models/churn_model.pkl`.*

### Step 4: Launch the Streamlit Interactive Dashboard
```powershell
streamlit run app.py
```
*Opens an interactive web browser dashboard featuring real-time KPI metrics, Plotly retention visualizations, and a live churn risk predictor.*

---

## 💡 Summary of Key Findings

### 1. A/B Testing Conclusion (Level 30 vs Level 40 Gate)
- **1-Day Retention (D1)**: Slightly higher at Level 30 (**44.82%**) than Level 40 (**44.23%**).
- **7-Day Retention (D7)**: Higher at Level 30 (**19.02%**) than Level 40 (**18.20%**).
- **Statistical Significance**: A Chi-Square Test of Independence yields $\chi^2 = 9.9591$ and $p = 0.00160 < 0.05$.
- **Business Action**: **Keep the gate at Level 30.** Placing the gate earlier paces player progress and prevents long-term player exhaustion/burnout.

### 2. Machine Learning Churn Prediction
- **Model Accuracy**: **87.75%**
- **ROC-AUC Score**: **0.8927**
- **Top Feature Drivers**:
  1. `sum_gamerounds` (**86.78%**): Week 1 gameplay intensity is the predominant predictor of retention.
  2. `retention_1_flag` (**13.15%**): Returning on Day 1 is a strong early indicator of Day 7 retention.
  3. `gate_45_flag` (**0.07%**): Gate assignment plays a subtle role relative to overall player engagement.

---

## 🛠️ Project Structure

```
mobile-gaming-churn-analytics/
├── data/
│   └── cookie_cats.csv         # Kaggle Cookie Cats dataset
├── models/
│   └── churn_model.pkl         # Saved Random Forest model artifact
├── .gitignore                  # Ignores venv, __pycache__, .pkl, csv
├── check_data.py               # Data integrity validation script
├── eda_and_abtest.py           # EDA & Chi-Square A/B test pipeline
├── churn_model.py              # ML model training & evaluation script
├── app.py                      # Streamlit interactive dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

Snapshots-
<img width="1908" height="892" alt="image" src="https://github.com/user-attachments/assets/332e2086-bd53-4b73-9762-c502929f829d" />
<img width="1377" height="737" alt="image" src="https://github.com/user-attachments/assets/527d17c0-b2d3-45c2-8e77-e830a4f5b023" />
<img width="1217" height="401" alt="image" src="https://github.com/user-attachments/assets/e2a979e6-c7e2-4b65-a645-0d98f3f0a084" />
<img width="1253" height="801" alt="image" src="https://github.com/user-attachments/assets/a421f764-9404-48c5-8c0d-f60e73c7fcd5" />
<img width="967" height="810" alt="image" src="https://github.com/user-attachments/assets/833c830f-82cc-4b26-b23a-f744b701cc2c" />
<img width="1127" height="740" alt="image" src="https://github.com/user-attachments/assets/77305773-a39b-4b5f-a46a-a7270cb3f92f" />







