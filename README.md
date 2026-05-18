#  Kenya Car Import Tool

A data science project that helps Kenyan car buyers estimate the total 
cost of importing a car from Japan, predict prices using Machine Learning, 
and search real Japanese car listings — all through an interactive dashboard.

---

##  Project Overview

This tool scrapes real car listings from carfromjapan.com, calculates the 
full KRA import cost breakdown, and predicts Japanese market prices using 
an XGBoost ML model. Built as a capstone data science project.

---

## Project Structure

| File | Description |
|------|-------------|
| `Capstone_project.ipynb` | Main notebook — scraping, cleaning, ML |
| `app.py` | Streamlit interactive dashboard |
| `car_listings.csv` | Raw scraped data (250 cars) |
| `car_listings_cleaned.csv` | Cleaned and processed data |
| `car_listings_with_costs.csv` | Data with full import cost calculations |
| `xgboost_model.pkl` | Trained XGBoost price prediction model |
| `model_columns.pkl` | Saved model feature columns |

---

##  How It Works

**Phase 1 — Scraping:** 250 Toyota Vitz listings scraped from 
carfromjapan.com using `requests` and `BeautifulSoup` across 10 pages.

**Phase 2 — Cleaning:** Raw strings converted to numeric types, 
registration year extracted, car age engineered, columns standardised.

**Phase 3 — Import Calculator:** Full KRA tax breakdown calculated 
for every car including Import Duty (35%), Excise Duty (20-35% by 
engine size), VAT (16%), IDF (3.5%), RDL (2%), plus port, clearing 
and registration fees.

**Phase 5 — ML Model:** XGBoost regressor trained on 7 features 
achieving R² of 0.78 and MAE of $3,037 on 250 listings.

**Phase 6 — Dashboard:** Streamlit web app with three tabs — 
Car Search, Import Cost Calculator, and Price Prediction.

---

## Model Performance

| Model | MAE | R² |
|-------|-----|----|
| Linear Regression | $4,031 | 0.60 |
| Random Forest | $3,003 | 0.77 |
| **XGBoost** | **$3,037** | **0.78** |

---

## Import Cost Formula
CIF = FOB (USD→KES) + Shipping (KES 130,000) + Insurance (1.5%)
KRA Taxes = Import Duty + Excise Duty + VAT + IDF + RDL
Total = CIF + KRA Taxes + Port (50K) + Clearing (22.5K) + Registration (10K)
---

##  How to Run

```bash
# Install dependencies
pip install requests beautifulsoup4 pandas scikit-learn xgboost joblib streamlit

# Run the dashboard
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.


---

## Built With

`Python` `BeautifulSoup4` `Pandas` `Scikit-learn` `XGBoost` `Streamlit`

---

## Results

> 250 cars scraped · 78% ML accuracy · Full KRA breakdown · 
> Live interactive dashboard
