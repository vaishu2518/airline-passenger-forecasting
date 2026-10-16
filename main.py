import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the airline passenger dataset
url = (
    "https://raw.githubusercontent.com/"
    "jbrownlee/Datasets/master/airline-passengers.csv"
)

df = pd.read_csv(url)

# Step 2: Display the first five rows
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Check the dataset information
print("\nDataset information:")
print(df.info())

# Step 4: Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Step 5: Display basic statistics
print("\nPassenger statistics:")
print(df["Passengers"].describe())

# Step 6: Plot historical passenger numbers
df["Month"] = pd.to_datetime(df["Month"])

plt.figure(figsize=(12, 5))
plt.plot(
    df["Month"],
    df["Passengers"],
    marker="o",
    linewidth=2
)

plt.title("Monthly Airline Passenger Numbers")
plt.xlabel("Month")
plt.ylabel("Number of Passengers (thousands)")
plt.grid(True)
plt.tight_layout()
plt.show()
from prophet import Prophet

# Step 7: Prepare data for Prophet
forecast_df = df[["Month", "Passengers"]].copy()
forecast_df.columns = ["ds", "y"]

# Step 8: Create and train the forecasting model
model = Prophet()
model.fit(forecast_df)

# Step 9: Create future dates for the next 24 months
future = model.make_future_dataframe(periods=24, freq="MS")

# Step 10: Generate predictions
forecast = model.predict(future)

# Step 11: Display the next 24 months of predictions
print("\nPassenger Forecast for the Next 24 Months:")
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(24))

# Step 12: Plot the forecast
fig = model.plot(forecast)
fig.savefig("airline_passenger_forecast.png", dpi=300, bbox_inches="tight")
plt.title("Airline Passenger Forecast - Next 24 Months")
plt.xlabel("Date")
plt.ylabel("Passengers (thousands)")
plt.tight_layout()
plt.show()
# Step 13: Evaluate forecast accuracy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Use the last 24 months as test data
train_df = forecast_df.iloc[:-24].copy()
test_df = forecast_df.iloc[-24:].copy()

# Train a new model using only the training data
evaluation_model = Prophet()
evaluation_model.fit(train_df)

# Predict the dates in the test period
test_future = test_df[["ds"]].copy()
test_predictions = evaluation_model.predict(test_future)

# Compare actual and predicted passenger numbers
actual = test_df["y"].values
predicted = test_predictions["yhat"].values

mae = mean_absolute_error(actual, predicted)
rmse = sqrt(mean_squared_error(actual, predicted))

print("\nFORECAST ACCURACY")
print("-----------------")
print(f"MAE:  {mae:.2f} thousand passengers")
print(f"RMSE: {rmse:.2f} thousand passengers")

# Step 14: Compare actual vs predicted passengers

plt.figure(figsize=(12, 5))

plt.plot(
    test_df["ds"],
    actual,
    marker="o",
    label="Actual Passengers"
)

plt.plot(
    test_df["ds"],
    predicted,
    marker="o",
    label="Predicted Passengers"
)

plt.title("Actual vs Predicted Airline Passengers")
plt.xlabel("Month")
plt.ylabel("Passengers (thousands)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()