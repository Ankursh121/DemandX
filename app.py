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

model = train_model(df)

# -------------------------------
# UI
# -------------------------------
st.title("🚀 Retail Demand Prediction AI")

# Sidebar
st.sidebar.header("Add Data")
sales = st.sidebar.number_input("Sales", 0)
marketing = st.sidebar.number_input("Marketing", 0)
discount = st.sidebar.number_input("Discount", 0)

if st.sidebar.button("Save"):
    save_data(sales, marketing, discount)
    st.success("Saved!")

# Inputs
st.sidebar.header("Prediction")
m = st.sidebar.slider("Marketing", 0, 500, 100)
d = st.sidebar.slider("Discount", 0, 50, 10)
day = st.sidebar.slider("Day", 0, 6, 2)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Dashboard", "Prediction", "Insights"])

# Dashboard
with tab1:
    st.subheader("Sales Trend")
    fig = px.line(df, x="date", y="sales")
    st.plotly_chart(fig)

# Prediction
with tab2:
    if st.button("Predict"):
        pred = predict(model, df, m, d, day)
        st.metric("Predicted Sales", int(pred))

        fig, ax = plt.subplots()
        ax.plot(df['day_number'], df['sales'])
        ax.scatter(df['day_number'].max() + 1, pred)
        st.pyplot(fig)

# Insights
with tab3:
    st.write(generate_insight(df))