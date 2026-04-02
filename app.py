import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from backend.data_handler import load_data, save_data, feature_engineering
from ml.model import train_model, predict
from utils.insights import generate_insight

st.set_page_config(layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
df = load_data()
df = feature_engineering(df)

# Sort data for proper graph
df = df.sort_values("date")

model = train_model(df)

# -------------------------------
# UI
# -------------------------------
st.title("🚀 Retail Demand Prediction AI")
st.markdown("### Smart Sales Forecasting using Machine Learning")

# -------------------------------
# SIDEBAR (ADD DATA)
# -------------------------------
st.sidebar.header("➕ Add Data")

sales = st.sidebar.number_input("Sales", min_value=0)
marketing = st.sidebar.number_input("Marketing Spend", min_value=0)
discount = st.sidebar.number_input("Discount (%)", min_value=0)

if st.sidebar.button("💾 Save Data"):
    save_data(sales, marketing, discount)
    st.success("Data saved successfully!")
    st.rerun()   # 🔥 auto refresh

# -------------------------------
# SIDEBAR (PREDICTION INPUT)
# -------------------------------
st.sidebar.header("📥 Prediction Input")

m = st.sidebar.slider("Marketing Spend", 0, 500, 100)
d = st.sidebar.slider("Discount (%)", 0, 50, 10)
day = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Prediction", "💡 Insights"])

# -------------------------------
# DASHBOARD
# -------------------------------
with tab1:
    st.subheader("📈 Sales Trend (Line Graph)")

    fig = px.line(df, x="date", y="sales", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📂 Recent Data")
    st.dataframe(df.tail(), use_container_width=True)

# -------------------------------
# PREDICTION
# -------------------------------
with tab2:
    st.subheader("🔮 Predict Future Sales")

    if st.button("🚀 Predict"):
        pred = predict(model, df, m, d, day)

        st.metric("Predicted Sales", int(pred))

        # -------------------------------
        # 🔥 CLEAN LINE GRAPH
        # -------------------------------
        fig2, ax = plt.subplots(figsize=(10, 5))

        # Line graph for actual data
        ax.plot(df['day_number'], df['sales'],
                label='Actual Sales',
                marker='o',
                linewidth=2)

        # Prediction point
        ax.scatter(df['day_number'].max() + 1, pred,
                   label='Predicted',
                   s=150,
                   marker='X')

        # Labels
        ax.set_xlabel("Day Number")
        ax.set_ylabel("Sales")
        ax.set_title("📈 Actual vs Predicted Sales")

        # Grid for readability
        ax.grid(True, linestyle='--', alpha=0.6)

        # Legend
        ax.legend()

        # Annotate prediction value
        ax.annotate(f"{int(pred)}",
                    (df['day_number'].max() + 1, pred),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')

        plt.tight_layout()

        st.pyplot(fig2)

# -------------------------------
# INSIGHTS
# -------------------------------
with tab3:
    st.subheader("💡 Business Insights")
    st.write(generate_insight(df))