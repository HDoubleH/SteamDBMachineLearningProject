# Multi-Model Steam Market Analysis: Predicting Game Success, Revenue, and Market Segments

## Overview

This project is a comprehensive data science and machine learning investigation into the factors that drive success within the highly competitive Steam PC game marketplace. Utilizing a dataset of over 111,000 games, the primary goal is to build **multiple predictive models** to inform development and publishing strategy.

---

## Project Goals

The analysis and modeling phase is structured around predicting **multiple target outcomes** using different machine learning approaches:

1.  **Classification:** Predict if a game will fall into a specific **Success Tier** (e.g., Failure, Modest Hit, Blockbuster).
2.  **Regression:** Predict the **Log-transformed Estimated Revenue** or **Estimated Owners**.
3.  **Clustering:** Identify natural **Market Segments** (niches) based on game features, playtime, and price.

---

## Data Source and EDA Highlights

* **Dataset:** Steam Games Dataset (sourced from Kaggle).
* **Initial Cleaning:** Custom parsing was required to handle structural issues within the source CSV.
* **Key Discovery:** The market exhibits extreme **outliers** across all success metrics. The low median estimated revenue confirms the market is dominated by a few massive outliers.
* **Feature Engineering:** Focus will be on creating normalized quality scores (Review Ratio) and transforming highly skewed variables (Log-Positive Reviews, Log-Revenue) for modeling.
* 
---

## Key Technologies

* **Language:** Python
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (for Classification, Regression, and Clustering)

---

