Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Pest Population Forecasting

![Python](https://img.shields.io/badge/python-3.8%2B-yellow)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/streamlit-deployed-red)

Dual-task ML pipeline predicting insect population counts (regression) and catch-event occurrence (classification) from meteorological and historical entomological data. Built to replace manual field observation with automated risk estimation for field managers.

## Live Demo

The app is deployed on HuggingFace Spaces — field teams can query real-time pest risk estimates from current weather inputs without any manual data handling:

**https://huggingface.co/spaces/parhamaki/Pest_Population_Forecasting**

## Problem

Field managers traditionally estimated pest risk through manual observation — slow, expensive, and weather-dependent. This project builds two predictive pipelines from sensor data:

- **Regression**: predict daily insect population counts
- **Classification**: predict whether a catch event will occur (binary)

Feature importance reports make predictions explainable to non-technical stakeholders.

## Approach

### Data & Feature Engineering

- Merged meteorological sensor readings with historical entomological catch records
- Engineered lag features to capture delayed weather-pest relationships
- Full preprocessing pipeline: imputation, scaling, encoding

### Models Compared

**Regression** (evaluated on MAE):
- Random Forest, Gradient Boosting, Ridge, SVR, XGBoost, LightGBM

**Classification** (evaluated on F1):
- Random Forest, Gradient Boosting, Logistic Regression, SVM, XGBoost, LightGBM

Best models selected through stratified cross-validation. Feature importance reports generated for all final models.

## Project Structure

```
├── notebooks/
│   ├── Notebook_1_Data_Preprocessing_&_EDA.ipynb
│   ├── Notebook_2_Regression_Modeling.ipynb
│   └── Notebook_3_Classification_Modeling.ipynb
├── models/                        # Serialised model artifacts
├── utils_eda.py                   # EDA helper functions
├── utils_regression.py            # Regression pipeline utilities
├── utils_classification.py        # Classification pipeline utilities
├── app.py                         # Streamlit application
├── cleaned_engineered_data.csv    # Feature-engineered dataset
├── requirements.txt
└── README.md
```

## Local Setup

```bash
git clone https://github.com/Alirezakhoshsolat/pest-population-forecasting.git
cd pest-population-forecasting
pip install -r requirements.txt
streamlit run app.py
```

## Course

Information Systems & Business Intelligence — University of Naples Federico II
Grade: 30/30

## License

MIT License
