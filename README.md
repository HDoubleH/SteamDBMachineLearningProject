# Steam Game Success Prediction: A Classification Analysis

## Project Goal

This project aims to build a robust machine learning classification model to **predict the binary success** of paid games on the Steam platform. Success is strictly defined by dual engagement thresholds: **Total Reviews ≥ 100 AND Average Playtime ≥ 60 minutes**.

The core challenge involves overcoming severe class imbalance and data leakage using only features available at or shortly after a game's launch.

***

## Project Structure

This repository contains two main Jupyter Notebooks that separate the Exploratory Data Analysis (EDA) from the Machine Learning workflow, adhering to documentation best practices.

| File | Description |
| :--- | :--- |
| `SteamDbProject.ipynb` | **Exploratory Data Analysis (EDA) & Preprocessing.** Covers data loading, complex row repair and cleaning, data exploration, visualizations (e.g., price distribution, correlation heatmap), feature selection, and justification for filtering to paid games. |
| `SteamDbProjectModeling.ipynb` | **Basic Modeling & Evaluation.** Covers feature scaling, stratified train-test split, training and comparison of tree-based models (Decision Tree, Random Forest, Gradient Boosting), and detailed analysis of performance metrics. |
| `README.md` | This document, providing a summary of project goals, key findings, limitations, and future work. |

***

## Key Findings

### EDA Insights

1.  **Market Skew:** The marketplace is dominated by low-cost and free-to-play games ($20.86\%$ free), which justified filtering the analysis to **paid games only** to create a more focused model.
2.  **Class Imbalance:** The target variable is severely imbalanced, with only approximately **$10\%$** of games classified as "Successful."
3.  **Data Leakage Identified:** Features like `Recommendations` and `Median playtime forever` showed critically high correlation with the target components and were **removed** to ensure the model makes valid predictions.

### Model Performance

* **Best Model:** The **Gradient Boosting Classifier** was selected as the best model, demonstrating the optimal balance between Precision and Recall.
* **F1 Score:** **0.55**, the primary metric for imbalanced data.
* **Precision (75.7%):** The model is reliable; when it predicts a game will be successful, it is correct over three-quarters of the time.
* **Recall (42.8%):** This is the main limitation. The model fails to capture over half of the truly successful games (high False Negative rate).

***

## Limitations and Future Work

### Primary Limitations

1.  **Low Recall:** The model's inability to identify a majority of successful games is the most significant constraint, driven by the complexities of predicting long-term engagement with early-stage features.
2.  **Feature Restriction:** The model is constrained to using less-predictive, early-stage features due to the necessary removal of highly predictive but leaky metrics.
3.  **Class Imbalance:** The $90/10$ class split introduces inherent bias toward the dominant class.

### Future Improvements

1.  **Optimization:** Implement systematic **Hyperparameter Tuning (e.g., Grid Search)** and **Classification Threshold Optimization** to maximize the F1 score.
2.  **Advanced Data Handling:** Apply techniques like setting **Class Weights** or using **SMOTE** (Synthetic Minority Over-sampling Technique) to the training phase to address the imbalance directly.
3.  **Feature Engineering:** The most critical step is to derive **non-leaky signals** from the previously removed variables (like `Recommendations`) to leverage their predictive power without compromising model validity.
4.  **Analysis:** Conduct and report a formal **Feature Importance** analysis to explain which features are most critical to the Gradient Boosting model's predictions.

***

## Setup and Dependencies

To run the notebooks, you will need the following Python libraries:

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
kagglehub
