import streamlit as st
import pandas as pd
import joblib
import os

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Car Import Tool",
    layout="wide"
)

st.title(" Kenya Car Import Tool")
st.markdown("Search Japanese cars, calculate import costs, and predict prices.")

# ── Load Data & Model ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, "car_listings_with_costs.csv")
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    model   = joblib.load(os.path.join(BASE_DIR, "xgboost_model.pkl"))
    columns = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))
    return model, columns

df             = load_data()
model, columns = load_model()

# ── Constants ──────────────────────────────────────────
USD_TO_KES        = 129.50
SHIPPING_COST_KES = 130_000
INSURANCE_RATE    = 0.015
IMPORT_DUTY_RATE  = 0.35
IDF_RATE          = 0.035
RDL_RATE          = 0.02
VAT_RATE          = 0.16
PORT_CHARGES_KES  = 50_000
CLEARING_FEES_KES = 22_500
REGISTRATION_KES  = 10_000

# ── Helper Functions ───────────────────────────────────
def get_excise_rate(engine_cc):
    if engine_cc <= 1000:
        return 0.20
    elif engine_cc <= 2000:
        return 0.25
    elif engine_cc <= 3000:
        return 0.30
    else:
        return 0.35

def calculate_import_cost(fob_usd, engine_cc):
    fob_kes       = fob_usd * USD_TO_KES
    insurance_kes = fob_kes * INSURANCE_RATE
    cif_kes       = fob_kes + SHIPPING_COST_KES + insurance_kes
    import_duty   = cif_kes * IMPORT_DUTY_RATE
    excise_duty   = cif_kes * get_excise_rate(engine_cc)
    idf           = cif_kes * IDF_RATE
    rdl           = cif_kes * RDL_RATE
    vat           = (cif_kes + import_duty + excise_duty) * VAT_RATE
    total_taxes   = import_duty + excise_duty + idf + rdl + vat
    other_charges = PORT_CHARGES_KES + CLEARING_FEES_KES + REGISTRATION_KES
    total_cost    = cif_kes + total_taxes + other_charges

    return {
        "FOB (KES)":           round(fob_kes, 2),
        "Shipping (KES)":      SHIPPING_COST_KES,
        "Insurance (KES)":     round(insurance_kes, 2),
        "CIF (KES)":           round(cif_kes, 2),
        "Import Duty (KES)":   round(import_duty, 2),
        "Excise Duty (KES)":   round(excise_duty, 2),
        "IDF (KES)":           round(idf, 2),
        "RDL (KES)":           round(rdl, 2),
        "VAT (KES)":           round(vat, 2),
        "Port Charges (KES)":  PORT_CHARGES_KES,
        "Clearing Fees (KES)": CLEARING_FEES_KES,
        "Registration (KES)":  REGISTRATION_KES,
        "TOTAL COST (KES)":    round(total_cost, 2)
    }

# ── Tabs ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    " Car Search",
    " Import Cost Calculator",
    " Price Prediction"
])

# ════════════════════════════════════════════════════════
# TAB 1 — CAR SEARCH
# ════════════════════════════════════════════════════════
with tab1:
    st.header("Search Japanese Car Listings")

    col1, col2, col3 = st.columns(3)

    with col1:
        min_price = st.number_input("Min Price (USD)", value=0, step=500)
        max_price = st.number_input("Max Price (USD)", value=50000, step=500)

    with col2:
        min_year = st.number_input("Min Year", value=2018, step=1)
        max_year = st.number_input("Max Year", value=2024, step=1)

    with col3:
        fuel_options  = ["All"] + sorted(df["fuel_type"].dropna().unique().tolist())
        trans_options = ["All"] + sorted(df["transmission"].dropna().unique().tolist())
        fuel_filter   = st.selectbox("Fuel Type", fuel_options)
        trans_filter  = st.selectbox("Transmission", trans_options)

    # Apply filters
    filtered = df.copy()
    filtered = filtered[
        (filtered["price"] >= min_price) &
        (filtered["price"] <= max_price) &
        (filtered["registration_year"] >= min_year) &
        (filtered["registration_year"] <= max_year)
    ]
    if fuel_filter != "All":
        filtered = filtered[filtered["fuel_type"] == fuel_filter]
    if trans_filter != "All":
        filtered = filtered[filtered["transmission"] == trans_filter]

    st.markdown(f"**{len(filtered)} cars found**")
    st.dataframe(filtered[[
        "price", "registration_year", "mileage",
        "engine_capacity", "fuel_type", "transmission",
        "TOTAL COST (KES)"
    ]].reset_index(drop=True))

# ════════════════════════════════════════════════════════
# TAB 2 — IMPORT COST CALCULATOR
# ════════════════════════════════════════════════════════
with tab2:
    st.header("Import Cost Calculator")
    st.markdown("Enter car details to estimate the total cost of importing to Kenya.")

    col1, col2 = st.columns(2)

    with col1:
        fob_price = st.number_input("Car Price in Japan (USD)", value=8000, step=500)
        engine_cc = st.number_input("Engine Size (cc)", value=1000, step=100)

    with col2:
        st.markdown("**Current Rates**")
        st.info(f"USD to KES: {USD_TO_KES}")
        st.info(f"Shipping: KES {SHIPPING_COST_KES:,}")

    if st.button("Calculate Import Cost"):
        costs = calculate_import_cost(fob_price, engine_cc)
        st.success(f"Total Import Cost: KES {costs['TOTAL COST (KES)']:,.2f}")
        st.markdown("### Full Breakdown")
        cost_df = pd.DataFrame(costs.items(), columns=["Item", "Amount (KES)"])
        cost_df["Amount (KES)"] = cost_df["Amount (KES)"].apply(
            lambda x: f"KES {x:,.2f}"
        )
        st.table(cost_df)

# ════════════════════════════════════════════════════════
# TAB 3 — ML PRICE PREDICTION
# ════════════════════════════════════════════════════════
with tab3:
    st.header("ML Price Prediction")
    st.markdown("Predict the Japanese market price of a car based on its features.")

    col1, col2 = st.columns(2)

    with col1:
        pred_year     = st.number_input("Registration Year", value=2020, step=1)
        pred_mileage  = st.number_input("Mileage (km)", value=30000, step=1000)
        pred_engine   = st.number_input("Engine Capacity (cc)", value=1000, step=100)

    with col2:
        pred_fuel     = st.selectbox("Fuel Type",    ["Petrol", "Diesel", "Hybrid", "Other"])
        pred_trans    = st.selectbox("Transmission", ["Automatic", "Manual"])
        pred_drive    = st.selectbox("Drive Type",   ["2WD", "4WD", "-"])
        pred_steering = st.selectbox("Steering",     ["Right", "Left", "Other", "-"])

    if st.button("Predict Price"):
        sample = pd.DataFrame([{col: 0 for col in columns}])
        sample["registration_year"]              = pred_year
        sample["mileage"]                        = pred_mileage
        sample["engine_capacity"]                = pred_engine
        sample[f"fuel_type_{pred_fuel}"]         = 1
        sample[f"transmission_{pred_trans}"]     = 1
        sample[f"drive_type_{pred_drive}"]       = 1
        sample[f"steering_{pred_steering}"]      = 1

        predicted    = model.predict(sample)[0]
        import_costs = calculate_import_cost(predicted, pred_engine)

        st.success(f"Predicted Japanese Price: US$ {predicted:,.2f}")
        st.info(f"Estimated Total Import Cost to Kenya: KES {import_costs['TOTAL COST (KES)']:,.2f}")