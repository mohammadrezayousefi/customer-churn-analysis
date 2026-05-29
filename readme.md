# 💎 AI-Driven Customer Retention & Churn Prediction Platform

An enterprise-level decision support system built to predict customer churn in the telecommunications sector and provide dynamic, actionable retention strategies. 

This project goes beyond simple classification by integrating **Explainable AI (XAI)** and a **Dynamic Recommender System** to help marketing and retention teams take the Next Best Action (NBA) for each high-risk customer.

## 🚀 Core Features

* **Robust ML Pipeline:** Handles highly imbalanced datasets using `SMOTE` and normalizes skewed numerical features via `PowerTransformer` (Yeo-Johnson) for optimal Logistic Regression performance.
* **Explainable AI (SHAP):** Deconstructs the model's black-box predictions to identify the exact root causes of churn for every individual user.
* **Dynamic Next Best Action (NBA):** A smart recommender engine that maps identified churn risks to a diverse catalog of personalized marketing campaigns (e.g., tailored discounts, 5G upgrades, VOD subscriptions).
* **Batch Processing:** Capable of analyzing large CSV datasets in seconds, automatically categorizing thousands of users into risk tiers (Low, Medium, High) for bulk SMS marketing campaigns.
* **MLOps Feedback Loop:** Tracks the acceptance or rejection of suggested campaigns by operators, saving logs to a CSV file for future model retraining.
* **Premium X/UI:** A responsive, dark-themed Streamlit dashboard utilizing Glassmorphism design and interactive `Plotly` Gauge charts for rapid decision-making.

## 🛠️ Tech Stack

* **Machine Learning:** `scikit-learn`, `imbalanced-learn` (SMOTE)
* **Explainability:** `shap`
* **Data Processing:** `pandas`, `numpy`
* **Frontend & Visualization:** `streamlit`, `plotly`