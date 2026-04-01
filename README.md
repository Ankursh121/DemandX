# DemandX


# Retail Demand Prediction System

A machine learning-powered web app that predicts retail sales based on marketing spend and discount strategies.

##  Features

-  Add daily sales data
-  Data visualization
-  Predict future sales using ML
-  Feature engineering (day number, day of week)

## 🛠 Tech Stack

- Python
- Streamlit
- scikit-learn
- Pandas
- Matplotlib

##  Machine Learning

We use a Linear Regression model trained on:
- Marketing Spend
- Discount
- Day of Week
- Day Number

##  Project Structure

Retail-Demand-Prediction/
│
├── app.py
├── sales.csv
├── requirements.txt
├── README.md
│
├── data/
│   └── data_processing.py
│
├── model/
│   └── train_model.py
│
├── utils/
│   └── visualization.py
