# ESRB Wizard: Predicting Game Ratings & Player Engagement using Steam Game Data

**Contributors:** Cobean, Sarah · Khan, Obaid · Rangwala, Zaki · Riarh, Josh · Surjadhana, Aristo

This project explores how game metadata — such as tags, genres, pricing, and ESRB-like age ratings — can predict **player engagement (average playtime)** and reveal which features most influence a game's success.

Developed as part of **CP322: Machine Learning**, the project combines multiple datasets (Steam metadata, ESRB ratings, and API-sourced content descriptors) into a unified modeling pipeline.

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
| [Steam Achievement Stats](https://www.kaggle.com/datasets/patrickgendotti/steam-achievementstatscom-rankings) | Engagement features (achievement completion rates). |
| [IGDB API](https://api-docs.igdb.com/#age-rating-content-description) | Optional: Additional ESRB content descriptors via API. |

---

## 🧠 Methods

1. **Data Cleaning and Preprocessing**
   - Drop missing or zero playtime values.  
   - Normalize numeric columns (Price, Owners, Age).  
   - One-hot encode Tags, Genres, and Categories arrays.  
   - Merge datasets on game `Name` or `AppID`.

2. **Model Training**
   - **RandomForestRegressor** – Baseline model for playtime prediction.  
   - **XGBoost Regressor** – Advanced model with tuned hyperparameters.

3. **Model Evaluation**
   - Metrics: R², RMSE, MAE.  
   - Cross-validation for robustness.

4. **Explainability**
   - SHAP analysis to interpret feature importance and direction of influence.  
   - Visualize how features like price, age rating, and genre affect engagement.

---

RandomForestRegressor – Baseline model for playtime prediction.

XGBoost Regressor – Advanced model with tuned hyperparameters.

Model Evaluation

Metrics: R², RMSE, MAE.

Cross-validation for robustness.

Explainability

SHAP analysis to interpret feature importance and direction of influence.

Visualize how features like price, age rating, and genre affect engagement.

🧱 Project Structure
esrb-wizard/
├── data/
│   ├── raw/                  # Original datasets from Kaggle + API
│   ├── processed/            # Cleaned, merged datasets
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling_regression.ipynb          # Final version
│   ├── 03a_model_demo_colab.ipynb            # Working demo
│   └── 04_explainability.ipynb
│
├── src/
│   ├── train_baseline_model.py               # Script version of final model
│   ├── preprocess_utils.py
│   ├── shap_analysis.py
│   └── model_utils.py
│
├── results/
│   ├── metrics.csv
│   ├── shap_summary.png
│   └── feature_importance.png
│
├── reports/
│   ├── ieee_report.tex
│   ├── ieee_report.pdf
│   └── slides/
│
├── .gitignore
├── LICENSE.md
├── requirements.txt
└── README.md

```bash
# Clone the repository
git clone https://github.com/<your-username>/esrb-wizard.git
cd esrb-wizard

# (Optional) Create and activate a virtual environment
python3 -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
