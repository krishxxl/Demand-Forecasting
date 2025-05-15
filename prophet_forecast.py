import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('../data/sales.csv')
df.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

# Initialize and fit the model
model = Prophet()
model.fit(df)

# Predict next 60 days
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
plt.title("Sales Forecast")
plt.show()

# Plot components
model.plot_components(forecast)
plt.show()

# Save model and forecast
import joblib
joblib.dump(model, '../model/prophet_model.pkl')
forecast.to_csv('../data/forecast.csv', index=False)
