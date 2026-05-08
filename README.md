# 🚗 Car Price Prediction Using Machine Learning

A machine learning project that predicts the **on-road price of cars** in India using a scikit-learn pipeline with a Random Forest regressor.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Sample Prediction](#sample-prediction)
- [License](#license)

---

## 📖 Project Overview

This project builds an end-to-end ML pipeline to estimate the on-road price of a car based on its specifications and pricing components. The workflow covers:

1. Loading and exploring the dataset
2. Cleaning currency-formatted columns
3. Identifying numerical and categorical features
4. Building separate preprocessing pipelines for each feature type
5. Training a `RandomForestRegressor`
6. Evaluating the model with MAE and R² metrics
7. Predicting on-road prices for new car inputs

---

## 📊 Dataset

**File:** `Car data.csv`

| Column | Description |
|---|---|
| `Name` | Car model name |
| `Min Price (Lakh)` | Minimum variant price (in Lakhs) |
| `Max Price (Lakh)` | Maximum variant price (in Lakhs) |
| `Range (kmpl)` | Fuel efficiency (km per litre) |
| `CC` | Engine displacement (cubic centimetres) |
| `Seats` | Number of seats |
| `Variants` | Number of available variants |
| `Type` | Fuel type (Petrol / Diesel / etc.) |
| `Ex-Showroom Price` | Base price before taxes (INR) |
| `RTO` | Road tax charges (INR) |
| `Insurance` | Insurance cost (INR) |
| `Other` | Miscellaneous charges (INR) |
| `Onroad Price` *(target)* | Total on-road price (INR) |

- **Rows:** 21 car models
- **Missing values:** None

---

## 📁 Project Structure

```
Car-Price-Prediction-Using-ML/
├── Car data.csv                 # Dataset
├── Car_Price_Prediction.ipynb   # Main Jupyter Notebook
├── LICENSE
└── README.md
```

---

## ⚙️ ML Pipeline

### Preprocessing

| Feature Type | Columns | Transformation |
|---|---|---|
| Numerical | `Min Price`, `Max Price`, `Range`, `CC`, `Seats`, `Variants`, `Ex-Showroom Price`, `RTO`, `Insurance`, `Other` | Median imputation |
| Categorical | `Name`, `Type` | Most-frequent imputation → One-Hot Encoding |

### Model

```
Pipeline:
  ├── ColumnTransformer (preprocessor)
  │     ├── numeric_transformer  → SimpleImputer(strategy="median")
  │     └── categorical_transformer → SimpleImputer + OneHotEncoder
  └── RandomForestRegressor(n_estimators=200, random_state=42)
```

---

## 📈 Model Performance

Evaluated on a 20% held-out test set (80/20 split):

| Metric | Value |
|---|---|
| **MAE (INR)** | ₹87,644.17 |
| **R² Score** | 0.9212 |

### Actual vs Predicted Prices

| Actual Price (₹) | Predicted Price (₹) |
|---|---|
| 16,71,667 | 14,85,737.63 |
| 6,98,009 | 7,98,856.13 |
| 7,51,770 | 8,02,917.47 |
| 7,28,516 | 8,05,970.15 |
| 8,41,750 | 8,64,592.73 |

---

## 🛠️ Requirements

Install all dependencies with:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Visualisation |
| `scikit-learn` | ML pipeline, preprocessing, and modelling |
| `jupyter` | Running the notebook |

---

## ▶️ How to Run

### Option 1 — Google Colab (recommended)

1. Upload `Car data.csv` to your Colab session (`/content/Car data.csv`).
2. Open `Car_Price_Prediction.ipynb` in Colab.
3. Run all cells.

### Option 2 — Local Jupyter

1. Clone the repository:
   ```bash
   git clone https://github.com/technicalabinesh/Car-Price-Prediction-Using-ML.git
   cd Car-Price-Prediction-Using-ML
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn jupyter
   ```
3. Update the dataset path in the notebook from:
   ```python
   df = pd.read_csv("/content/Car data.csv")
   ```
   to:
   ```python
   df = pd.read_csv("Car data.csv")
   ```
4. Launch Jupyter and run all cells:
   ```bash
   jupyter notebook Car_Price_Prediction.ipynb
   ```

---

## 🔍 Sample Prediction

The notebook includes an example prediction for a **Hyundai Creta**:

```python
sample = {
    'Name': 'Hyundai Creta',
    'Min Price (Lakh)': 12,
    'Max Price (Lakh)': 18,
    'Range (kmpl)': 20,
    'CC': 1497,
    'Seats': 5,
    'Variants': 8,
    'Type': 'SUV',
    'Ex-Showroom Price': 1500000,
    'RTO': 150000,
    'Insurance': 60000,
    'Other': 20000
}
# Predicted Onroad Price: ₹ 1,524,104.64
```

---

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
