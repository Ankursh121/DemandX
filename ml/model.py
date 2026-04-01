from sklearn.linear_model import LinearRegression
import pandas as pd

def train_model(df):
    X = df[['day_number', 'day_of_week', 'marketing', 'discount']]
    y = df['sales']

    model = LinearRegression()
    model.fit(X, y)

    return model

def predict(model, df, marketing, discount, day):
    input_data = pd.DataFrame({
        'day_number': [df['day_number'].max() + 1],
        'day_of_week': [day],
        'marketing': [marketing],
        'discount': [discount]
    })

    return model.predict(input_data)[0]