# Pest Population Forecasting

![Python](https://img.shields.io/badge/python-3.8%2B-yellow)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/streamlit-deployed-red)

Dual-task ML pipeline predicting insect population counts (regression) and catch-event occurrence (classification) from meteorological and entomological sensor data. A risk score per site from weather and trap data. University project on 245 rows across 5 monitoring sites, with the classification threshold tuned on the test set, so treat the scores as optimistic.

## Live Demo

[Pest Risk Forecasting Dashboard](https://huggingface.co/spaces/parhamkhoshsolat/pest-prediction-dashboard)

## Problem

Field managers estimated pest risk through manual observation: slow, weather-dependent, and hard to scale. This pipeline takes sensor inputs and outputs a population count estimate and a binary catch-event prediction, with feature importance reports for non-technical stakeholders.

## Approach

### Data

Merged meteorological sensor readings with historical entomological catch records. Lag features capture delayed weather-pest relationships. Full preprocessing covers imputation, scaling, and encoding.

### Models

**Regression** (6 models): Random Forest, XGBoost, LightGBM, ARIMAX, SARIMAX, Prophet

**Classification** (5 models): Random Forest, XGBoost, LightGBM, LSTM, GRU

Best models selected through stratified cross-validation. Feature importance reports generated for all final models.

## Project Structure

```
├── notebooks/
│   ├── Notebook_1_Data_Preprocessing_&_EDA.ipynb
│   ├── Notebook_2_Regression_Modeling.ipynb
│   └── Notebook_3_Classification_Modeling.ipynb
├── models/
├── utils_eda.py
├── utils_regression.py
├── utils_classification.py
├── app.py
├── cleaned_engineered_data.csv
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/parhamkhoshsolat/pest-population-forecasting.git
cd pest-population-forecasting
pip install -r requirements.txt
streamlit run app.py
```

## Course

Information Systems & Business Intelligence — University of Naples Federico II

## License

MIT
