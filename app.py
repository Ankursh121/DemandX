import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Retail Demand Prediction System")
st.write("Predict sales using marketing and discount inputs")

# -------------------------------
# ADD NEW DATA
# -------------------------------
st.sidebar.subheader("➕ Add Daily Data")

new_sales = st.sidebar.number_input("Sales", min_value=0)
new_marketing = st.sidebar.number_input("Marketing Spend", min_value=0)
new_discount = st.sidebar.number_input("Discount (%)", min_value=0)

if st.sidebar.button("Save Data"):
    new_row = pd.DataFrame({
        'date': [pd.Timestamp.today().date()],
        'sales': [new_sales],
        'marketing': [new_marketing],
        'discount': [new_discount]
    })
    new_row.to_csv("sales.csv", mode='a', header=False, index=False)
    st.success("Data saved successfully!")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("sales.csv")

# Fix date format safely
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna()

st.subheader("📂 Dataset Preview")
st.write(df.tail())

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df['day_number'] = (df['date'] - df['date'].min()).dt.days
df['day_of_week'] = df['date'].dt.dayofweek

# -------------------------------
# MODEL TRAINING
# -------------------------------
X = df[['day_number', 'day_of_week', 'marketing', 'discount']]
y = df['sales']

model = LinearRegression()
model.fit(X, y)

# -------------------------------
# USER INPUT FOR PREDICTION
# -------------------------------
st.sidebar.header("📥 Prediction Input")

marketing_input = st.sidebar.slider("Marketing Spend", 0, 500, 100)
discount_input = st.sidebar.slider("Discount (%)", 0, 50, 10)
day_input = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        'day_number': [df['day_number'].max() + 1],
        'day_of_week': [day_input],
        'marketing': [marketing_input],
        'discount': [discount_input]
    })

    prediction = model.predict(input_data)

    st.subheader("📈 Predicted Sales")
    st.metric("Expected Sales", int(prediction[0]))

    # -------------------------------
    # GRAPH
    # -------------------------------
    fig, ax = plt.subplots()

    ax.plot(df['day_number'], df['sales'], label='Actual Sales')
    ax.scatter(df['day_number'].max() + 1, prediction[0], label='Predicted', marker='o')

    ax.set_xlabel("Day Number")
    ax.set_ylabel("Sales")
    ax.set_title("Sales Prediction")
    ax.legend()

    st.pyplot(fig)