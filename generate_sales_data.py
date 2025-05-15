import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Create output folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Generate date range for 2 years (730 days)
date_range = pd.date_range(start='2023-01-01', periods=730, freq='D')

# Components of sales
base_sales = 100
trend = np.linspace(0, 50, len(date_range))  # upward trend
seasonality = 20 * np.sin(2 * np.pi * date_range.dayofyear / 365.25)  # yearly seasonality
weekly = 10 * np.where(date_range.dayofweek >= 5, 1.5, 1)  # weekends spike
noise = np.random.normal(loc=0, scale=8, size=len(date_range))  # randomness

# Total sales = base + trend + seasonality + weekly + noise
sales = base_sales + trend + seasonality + weekly + noise
sales = np.maximum(sales, 0).astype(int)  # avoid negative sales

# Create DataFrame
df = pd.DataFrame({'date': date_range, 'sales': sales})

# Save to CSV
df.to_csv('data/sales.csv', index=False)

print("✅ Synthetic sales data generated and saved to 'data/sales.csv'")
