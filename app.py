import os
import pickle
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for modern card container styling
st.markdown(
    """
    <style>
    /* Global background and typography */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container styling */
    .header-box {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .header-title {
        background-color: #3b82f6;
        color: white;
        padding: 10px 28px;
        border-radius: 8px;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 1px;
        display: inline-block;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
    }
    .header-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 8px;
    }

    /* Custom Card Containers */
    div[data-testid="stColumn"]:nth-child(1) > div {
        background: #ffffff;
        border: 2px solid #0284c7;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    
    div[data-testid="stColumn"]:nth-child(2) > div {
        background: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }

    div[data-testid="stColumn"]:nth-child(3) > div {
        background: #ffffff;
        border: 2px solid #9333ea;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }

    /* Button Styling */
    .stButton>button {
        background: #0284c7;
        color: white;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        width: 100%;
        box-shadow: 0 4px 10px rgba(2, 132, 199, 0.3);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: #0369a1;
        transform: translateY(-1px);
        box-shadow: 0 6px 15px rgba(2, 132, 199, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 2. Load Model Pipeline
model_filename = "car_model.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(model_filename):
        st.error(
            f"⚠️ Model file '{model_filename}' not found! Run 'train_model.py' first."
        )
        st.stop()
    with open(model_filename, "rb") as file:
        return pickle.load(file)


model = load_model()

# 3. App Header Banner
st.markdown(
    """
    <div class="header-box">
        <div class="header-title">🚗 CAR PRICE PREDICTOR</div>
        <div class="header-subtitle">Fill in the specifications below to estimate the resale selling price</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# 4. Three-Column Card Layout
col1, col2, col3 = st.columns(3)

# --- COLUMN 1: CAR SPECIFICATIONS ---
with col1:
    st.subheader("🏎️ Car Details")
    st.caption("Manufacturing year and original showroom price.")

    year = st.number_input("Manufacturing Year", 2000, 2026, 2018, step=1)
    present_price = st.number_input(
        "Present Showroom Price (in Lakhs)", 0.1, 100.0, 6.5, step=0.1
    )
    driven_kms = st.number_input(
        "Total Kilometers Driven", 0, 1000000, 30000, step=1000
    )

# --- COLUMN 2: VEHICLE CONFIGURATION ---
with col2:
    st.subheader("⚙️ Configuration")
    st.caption("Fuel, transmission, and ownership details.")

    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
    owner = st.selectbox(
        "Number of Previous Owners",
        [0, 1, 2, 3],
        format_func=lambda x: "0 (First Owner)" if x == 0 else f"{x} Owner(s)",
    )

# --- COLUMN 3: PREDICTION ENGINE ---
with col3:
    st.subheader("📊 Price Prediction")
    st.caption("Click below to estimate the expected selling price.")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("💰 Predict Selling Price")

    if predict_clicked:
        car_age = 2026 - year

        input_data = pd.DataFrame(
            [
                {
                    "Present_Price": present_price,
                    "Driven_kms": driven_kms,
                    "Fuel_Type": fuel_type,
                    "Selling_type": selling_type,
                    "Transmission": transmission,
                    "Owner": owner,
                    "Car_Age": car_age,
                }
            ]
        )

        try:
            prediction = model.predict(input_data)[0]
            final_price = max(0.0, prediction)

            st.markdown("---")
            st.metric(
                label="Estimated Selling Price",
                value=f"₹ {final_price:.2f} Lakhs",
            )

            depreciation = (
                ((present_price - final_price) / present_price) * 100
                if present_price > 0
                else 0
            )
            st.caption(f"Estimated Depreciation: **{depreciation:.1f}%**")
            st.balloons()

        except Exception as e:
            st.error(f"Prediction Error: {e}")