from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv("4.csv")

# Separate features (X) and target variable (y)
X = dataset.drop("Fare", axis=1)
y = dataset["Fare"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but can be beneficial for linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Render the home page with a form
@app.route('/')
def home():
    return render_template('index.html', predicted_fare=None)

# Handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input values
    distance = float(request.form['distance'])
    duration = int(request.form['duration'])
    passengers = int(request.form['passengers'])
    time_of_day = int(request.form['time_of_day'])
    weather_condition = int(request.form['weather_condition'])
    traffic_condition = int(request.form['traffic_condition'])

    # Create a DataFrame with the user input
    user_data = pd.DataFrame({
        'Distance': [distance],
        'Duration': [duration],
        'Passengers': [passengers],
        'TimeOfDay': [time_of_day],
        'WeatherCondition': [weather_condition],
        'TrafficCondition': [traffic_condition]
    })

    # Standardize the user input
    user_data_scaled = scaler.transform(user_data)

    # Make a prediction using the trained model
    predicted_fare = model.predict(user_data_scaled)[0]

    # Render the home page with the result
    return render_template('index.html', predicted_fare=predicted_fare)

if __name__ == '__main__':
    app.run(debug=True)
