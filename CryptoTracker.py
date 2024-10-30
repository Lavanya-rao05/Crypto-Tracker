from binance.client import Client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Initialize the Binance client
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
client = Client(api_key, api_secret)

def get_historical_data(symbol, interval='1d', limit=100):
    """Fetch historical candlestick data."""
    klines = client.get_historical_klines(symbol, interval, limit=limit)
    df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                        'Close Time', 'Quote Asset Volume', 'Number of Trades', 
                                        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
    df['Close'] = df['Close'].astype(float)  # Convert to float
    return df[['Close']]

# Fetch historical data for Bitcoin
historical_data = get_historical_data('BTCUSDT')
print(historical_data.head())

# Prepare the Data
historical_data['Prev Close'] = historical_data['Close'].shift(1)  # Previous day's closing price
historical_data = historical_data.dropna()  # Drop rows with NaN values

# Features and target variable
X = historical_data[['Prev Close']]
y = historical_data['Close']

# Train the Model
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Compare predictions with actual prices
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(results.head())

# Evaluate Model Performance
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Visualize Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, predictions, label='Predicted', color='orange')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.show()
