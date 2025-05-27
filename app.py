import streamlit as st
import joblib
import numpy as np

# Load model
early_model = joblib.load('models/early_stage_model.pkl')
growth_model = joblib.load('models/expanded_startup_profit_model.pkl')

st.title("🌍 Startup Profit Predictor 💼")

# Country and currency mapping (updated)
currency_rates = {
    "India": {"symbol": "₹", "to_usd": 1/83},
    "USA": {"symbol": "$", "to_usd": 1.0},
    "UK": {"symbol": "£", "to_usd": 1/0.78},
    "EU": {"symbol": "€", "to_usd": 1/0.92},
    "Germany": {"symbol": "€", "to_usd": 1/0.92},  # Same as EU
    "Australia": {"symbol": "A$", "to_usd": 1/1.51}  # 1 USD = 1.51 AUD → 1 AUD = 1/1.51 USD
}

# 1. Country Selection
country = st.selectbox("Select Your Country", list(currency_rates.keys()))
currency = currency_rates[country]["symbol"]
conversion_rate = currency_rates[country]["to_usd"]

st.subheader(f"Enter Costs in your Local Currency ({currency})")

# 2. Cost Inputs
rd_spend = st.number_input("R&D Spend", min_value=0.0 , help="Amount spent on research and development")
admin = st.number_input("Administration Spend", min_value=0.0 , help="Administrative overhead costs")
marketing = st.number_input("Marketing Spend", min_value=0.0 , help="Marketing and advertising expenses")

# 3. Additional Feature Inputs
st.subheader("Enter Additional Startup Details")
startup_age = st.number_input("Startup Age", min_value=0 , value=0 , help="Years since the startup began. Use 0 for new ideas.")
st.caption("Set '0' if you're in the idea or pre-launch phase of your startup.")
funding = st.number_input(f"Funding Amount ({currency})", min_value=0.0)
experience = st.number_input("Founder Experience (years)", min_value=0)
industry = st.selectbox("Industry Area", ["Tech", "Healthcare", "Finance", "Education", "E-commerce"])
employees = st.number_input("Number of Employees", min_value=0)

# One-hot encode industry (example with 5 options)
industry_encoded = [0]*4
if industry == "Healthcare":
    industry_encoded[0] = 1
elif industry == "Finance":
    industry_encoded[1] = 1
elif industry == "Education":
    industry_encoded[2] = 1
elif industry == "E-commerce":
    industry_encoded[3] = 1
# "Tech" is baseline (all zeros)

st.markdown("""
    <style>
        .centered-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .stButton > button {
            background: linear-gradient(90deg, #ff69b4, #8a2be2);
            color: white;
            font-weight: bold;
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #ff85c1, #9b30ff);
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Custom button style
st.markdown("""
    <style>
        .centered-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .stButton > button {
            background: linear-gradient(90deg, #ff69b4, #8a2be2);
            color: white;
            font-weight: bold;
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #ff85c1, #9b30ff);
            transform: scale(1.05);
            color: purple;
        }
    </style>
""", unsafe_allow_html=True)

# Centered styled button
st.markdown('<div class="centered-button">', unsafe_allow_html=True)
predict = st.button("Predict Profit")
st.markdown('</div>', unsafe_allow_html=True)

# Prediction logic
if predict:
    # Convert all currency-based features to USD
    rd_spend_usd = rd_spend * conversion_rate
    admin_usd = admin * conversion_rate
    marketing_usd = marketing * conversion_rate
    funding_usd = funding * conversion_rate

    # Input feature order: R&D, Admin, Marketing, Startup Age, Funding, Experience, Employees, + 4 industry flags
    input_data = np.array([[rd_spend_usd, admin_usd, marketing_usd,
                            startup_age, funding_usd, experience, employees] + industry_encoded])

    if startup_age == 0:
        prediction_usd = early_model.predict(input_data)[0]
    else:
        prediction_usd = growth_model.predict(input_data)[0]

    prediction_local = prediction_usd / conversion_rate

    st.success(f"Predicted 1-Year Profit: {currency}{prediction_local:,.2f}")
    st.caption(f"(* Model uses USD internally. Conversion rate: 1 {currency} = {conversion_rate:.4f} USD)")

