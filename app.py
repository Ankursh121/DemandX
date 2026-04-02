import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from backend.data_handler import load_data, save_data, feature_engineering
from ml.model import train_model, predict
from utils.insights import generate_insight

st.set_page_config(layout="wide")

# -------------------------------
# ₹ FORMAT FUNCTION
# -------------------------------
def format_inr(value):
    return f"₹{int(value):,}"

# -------------------------------
# LOAD DATA
# -------------------------------
df = load_data()
df = feature_engineering(df)
df = df.sort_values("date")

model = train_model(df)

# -------------------------------
# HEADER (BRANDING)
# -------------------------------
st.markdown("## 🚀 DemandX AI")
st.markdown("### Smart Demand Forecasting & Business Intelligence")
st.markdown("---")

# -------------------------------
# SIDEBAR - ADD DATA
# -------------------------------
st.sidebar.header("➕ Add Data")

sales = st.sidebar.number_input("Sales Units", min_value=0)
marketing = st.sidebar.number_input("Marketing Spend (₹)", min_value=0)
discount = st.sidebar.number_input("Discount (%)", min_value=0)

if st.sidebar.button("💾 Save Data"):
    save_data(sales, marketing, discount)
    st.success("Data saved successfully!")
    st.rerun()

# -------------------------------
# SIDEBAR - INPUT
# -------------------------------
st.sidebar.header("📥 Prediction Settings")

m = st.sidebar.slider("Marketing Spend (₹)", 0, 500, 100)
d = st.sidebar.slider("Discount (%)", 0, 50, 10)
day = st.sidebar.slider("Day (0=Mon)", 0, 6, 2)

forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)
price = st.sidebar.number_input("Product Price (₹)", value=100)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Prediction", "💡 Insights"])

# -------------------------------
# DASHBOARD
# -------------------------------
with tab1:
    st.subheader("📈 Sales Trend")

    fig = px.line(df, x="date", y="sales", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Avg Sales", f"{int(df['sales'].mean()):,}")
    col3.metric("Max Sales", f"{int(df['sales'].max()):,}")

    st.subheader("📂 Recent Data")
    st.dataframe(df.tail(), use_container_width=True)

# -------------------------------
# PREDICTION
# -------------------------------
with tab2:
    st.subheader("🔮 Future Demand Forecast")

    if st.button("🚀 Predict"):

        pred = predict(model, df, m, d, day)

        col1, col2 = st.columns(2)

        col1.metric("Next Day Sales", f"{int(pred):,}")

        revenue = pred * price
        col2.metric("Estimated Revenue", format_inr(revenue))

        # -------------------------------
        # FORECAST
        # -------------------------------
        future_preds = []
        future_days_list = []

        for i in range(forecast_days):
            day_number = df['day_number'].max() + i + 1

            input_data = pd.DataFrame({
                'day_number': [day_number],
                'day_of_week': [(day + i) % 7],
                'marketing': [m],
                'discount': [d]
            })

            p = model.predict(input_data)[0]
            future_preds.append(p)
            future_days_list.append(day_number)

        # -------------------------------
        # MODEL ACCURACY
        # -------------------------------
        X = df[['day_number','day_of_week','marketing','discount']]
        y = df['sales']

        mae = mean_absolute_error(y, model.predict(X))

        st.info(f"📊 Model MAE: {mae:.2f}")

        # -------------------------------
        # GRAPH (LINE FORMAT)
        # -------------------------------
        st.subheader("📈 Sales Forecast")

        fig2, ax = plt.subplots(figsize=(10, 5))

        ax.plot(df['day_number'], df['sales'],
                label='Actual Sales',
                marker='o',
                linewidth=2)

        ax.plot(future_days_list, future_preds,
                label='Forecast',
                linestyle='dashed',
                marker='o')

        ax.set_xlabel("Day Number")
        ax.set_ylabel("Sales")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        st.pyplot(fig2)

# -------------------------------
# INSIGHTS
# -------------------------------
with tab3:
    st.subheader("💡 Business Insights")
    st.write(generate_insight(df))