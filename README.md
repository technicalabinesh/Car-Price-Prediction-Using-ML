# Car Price Prediction Using Machine Learning

This project predicts **car on-road prices** using a machine learning pipeline built with scikit-learn.

## Project Overview

The notebook:
- Loads car data
- Cleans currency-formatted columns
- Splits numerical and categorical features
- Builds preprocessing + model pipeline
- Trains a Random Forest regressor
- Evaluates performance and predicts sample prices

Main artifact:
- `Car_Price_Prediction.ipynb`

## Model Summary

- **Target:** `Onroad Price`
- **Features:** `Name`, price range, mileage, engine CC, seats, variants, type, ex-showroom, RTO, insurance, other charges
- **Preprocessing:**
  - Median imputation for numeric features
  - Most-frequent imputation + one-hot encoding for categorical features
- **Regressor:** `RandomForestRegressor(n_estimators=200, random_state=42)`

Notebook evaluation output:
- **MAE:** `87644.17`
- **R² Score:** `0.9211971106843113`

## Requirements

Install Python dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

## How to Run

1. Open the notebook:
   - `Car_Price_Prediction.ipynb`
2. Ensure your dataset CSV is available and update the dataset path if needed.
   - The notebook currently reads: `"/content/Car data.csv"` (Google Colab style path)
3. Run all cells to train and evaluate the model.

## Notes

- This repository currently contains the notebook workflow.
- If you run locally (not Colab), change the dataset path to your local file location.
