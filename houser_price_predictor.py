# house_price_predictor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def main():
    # Load dataset
    house = pd.read_csv("https://raw.githubusercontent.com/ybifoundation/Dataset/main/Boston.csv")
    
    # Select features and target
    y = house['MEDV']
    x = house[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
               'PTRATIO', 'B', 'LSTAT']]
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=2529)
    
    # Train model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Predictions
    y_pred = model.predict(x_test)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Display results
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Absolute Percentage Error: {mape * 100:.2f}%")
    print(f"Mean Squared Error: {mse:.2f}")

if __name__ == "__main__":
    main()
