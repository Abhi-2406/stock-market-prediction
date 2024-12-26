import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt

# Load the newly generated dataset
data_path = data_path = r"C:\Big Dada analytics\balanced_stock_market_data.csv"
  # Replace with your downloaded dataset file path
data = pd.read_csv(data_path)

# Prepare features and target variable
X = data[["Open_Price", "Todays_Low", "Todays_High", "Prev_Close", "52_Week_Low", "52_Week_High", "Volume", "Avg_Price"]]
y = data["Is_Good_To_Buy"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
with open("stock_prediction_model_20000.pkl", "wb") as file:
    pickle.dump(model, file)

# Function to predict stock using user input
def predict_stock(open_price, todays_low, todays_high, prev_close, week_low, week_high, volume, avg_price):
    with open("stock_prediction_model_20000.pkl", "rb") as file:
        model = pickle.load(file)
    
    # Prepare input data
    input_data = np.array([[open_price, todays_low, todays_high, prev_close, week_low, week_high, volume, avg_price]])
    
    # Get prediction probabilities
    prediction = model.predict(input_data)[0]
    prediction_prob = model.predict_proba(input_data)[0]
    
    # Output results
    result = "Good to Buy" if prediction == 1 else "Not Good to Buy"
    print(f"The stock is: {result}")
    
    # Plot probabilities
    categories = ["Not Good to Buy", "Good to Buy"]
    plt.bar(categories, prediction_prob, color=['red', 'green'])
    plt.title("Stock Prediction Probabilities")
    plt.ylabel("Probability")
    plt.show()

# Take user input
print("Enter stock details for prediction:")
open_price = float(input("Open Price: "))
todays_low = float(input("Today's Low: "))
todays_high = float(input("Today's High: "))
prev_close = float(input("Previous Close: "))
week_low = float(input("52-Week Low: "))
week_high = float(input("52-Week High: "))
volume = int(input("Volume: "))
avg_price = float(input("Average Price: "))

# Predict stock
predict_stock(open_price, todays_low, todays_high, prev_close, week_low, week_high, volume, avg_price)
