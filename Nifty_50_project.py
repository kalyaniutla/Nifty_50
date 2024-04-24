import pandas  as pd
import matplotlib.pyplot as plt

nifty_data = pd.read_csv(r"E:\stock_market_courses\research\index-data\NIFTY 50_Historical_PR_01022022to13022024.csv")

a = pd.DataFrame(nifty_data)
print(a)
import pandas as pd

# Export to CSV
monthly_grouped.to_csv('monthly_grouped_data.csv')

print("Data exported to CSV successfully.")


# Create pivot table for Date, Open, Close
pivot_table_1 = pd.pivot_table(a, values=['Open', 'Close'], index='Date')

# Create pivot table for Date, High, Low
pivot_table_2 = pd.pivot_table(a, values=['High', 'Low'], index='Date')

# Display pivot tables
print("Pivot Table 1:")
print(pivot_table_1)

print("\n Pivot Table 2:")
print(pivot_table_2)

# Predicting prices for the coming 10 days can be done using various techniques such as
# time series forecasting models like ARIMA, Prophet, LSTM, etc. You need historical data and
# a chosen model to make predictions.
# Here's a basic example using linear regression for demonstration:

from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data for linear regression
X = np.array(range(len(a))).reshape(-1, 1)
y = np.array(a['Close'])

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict prices for the next 10 days
next_10_days = np.array(range(len(a), len(a) + 10)).reshape(-1, 1)
predicted_prices = model.predict(next_10_days)

# Display predicted prices
print("\nPredicted Prices for the Next 10 Days:")
for i, price in enumerate(predicted_prices):
    print(f"Day {i+1}: {price}")
# Combine existing and predicted data
predicted_dates = pd.date_range(start=a['Date'].iloc[-1], periods=10)
predicted_data = pd.DataFrame({
    'Date': predicted_dates.strftime('%d %b %Y'),
    'Close': predicted_prices
})

combined_data = pd.concat([a[['Date', 'Close']], predicted_data])

# Save combined data to CSV
combined_data.to_csv('combined_data.csv', index=False)

# Visualize pivot tables
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

pivot_table_1.plot(ax=axes[0], kind='bar', color=['skyblue', 'lightgreen'])
axes[0].set_title('Pivot Table 1')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price')

pivot_table_2.plot(ax=axes[1], kind='bar', color=['orange', 'lightblue'])
axes[1].set_title('Pivot Table 2')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Price')

plt.tight_layout()
plt.savefig('pivot_table_visualizations.png')
plt.show()
