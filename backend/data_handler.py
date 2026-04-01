import pandas as pd

def load_data():
    df = pd.read_csv("sales.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna()
    return df

def save_data(sales, marketing, discount):
    new_row = pd.DataFrame({
        'date': [pd.Timestamp.today().date()],
        'sales': [sales],
        'marketing': [marketing],
        'discount': [discount]
    })
    new_row.to_csv("sales.csv", mode='a', header=False, index=False)

def feature_engineering(df):
    df['day_number'] = (df['date'] - df['date'].min()).dt.days
    df['day_of_week'] = df['date'].dt.dayofweek
    return df