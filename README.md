# ESRB Wizard: Predicting Game Ratings & Player Engagement using Steam Game Data

**Contributors:** Cobean, Sarah · Khan, Obaid · Rangwala, Zaki · Riarh, Josh · Surjadhana, Aristo

This project explores how game metadata — such as tags, genres, pricing, and ESRB-like age ratings — can predict **player engagement (average playtime)** and reveal which features most influence a game's success.

Developed as part of **CP322: Machine Learning**, the project combines multiple datasets (Steam metadata & ESRB content ratings) into a unified modeling pipeline.

---

## 🧩 Objectives

1. **Predict average playtime** using Steam game features.
2. **Interpret key drivers** of engagement using SHAP explainability.
3. Optionally explore how ESRB maturity levels correlate with playtime.

---

## 📊 Datasets

| Source | Description |
|--------|--------------|
| [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) | Core dataset containing price, owners, playtime, tags, and genres. |
| [Video Games Rating by ESRB](https://www.kaggle.com/datasets/imohtn/video-games-rating-by-esrb) | ESRB maturity ratings for merging via game title or fuzzy match. |

---

## 🧠 Methods

1. **Steam Data Cleaning and Preprocessing**
   - Drop games missing a title or unique Steam ID numnber
   - Drop columns that will create noise or had strong multicolinearity
   - Simplified certain column values
   - Drop all non-english games
   - One-hot encode Genres, and Categories arrays
   - Drop missing values across all other features
  
2. **ESRB Data Cleaning and Preprocessing**
   - Drop rows with missing values
   - Drop ESRB age and console

3. **Classification**
   - **Dataset** - Cleansed Steam dataset
   - **RandomForestRegressor** – Trained on the basic dataset, a random undersample and random SMOTE oversample
   - **XGBoost Regressor** – Trained on the basic dataset, a random undersample and random SMOTE oversample
   - **LightGBM Regressor** – Trained on the basic dataset, a random undersample and random SMOTE oversample
   - **Performance Metrics** - PR-AUC, Precision, Recall, F1-Score
   - **Explainability** - SHAP analysis, visualize if mature themes affect engagement.

4. **Regression**
   - **Dataset** - ESRB dataset fuzzy-matched against subset of Steam dataset with non-zero playtime in past two weeks
   - **XGBoost Regressor** – Trained on the basic dataset, with IQR outliers removed (and each of these with log-transformed y)
   - **LightGBM Regressor** – Trained on the basic dataset, with IQR outliers removed (and each of these with log-transformed y)
   - **Performance Metrics** - MSE, RMSE, MAE, R^2
   - **Explainability** - SHAP analysis, visualize if mature themes affect engagement (before and after ablation)

---


## 🧱 Steps to Install

```bash
# Clone the repository
git clone https://github.com/<your-username>/esrb-wizard.git
cd esrb-wizard

# (Optional) Create and activate a virtual environment
python3 -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
