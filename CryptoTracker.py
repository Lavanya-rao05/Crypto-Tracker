from binance.client import Client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


api_key = 'YOUR_API_KEY' # add ur keys
api_secret = 'YOUR_API_SECRET'
client = Client(api_key, api_secret)

def get_historical_data(symbol, interval='1d', limit=100):
    """Fetch historical candlestick data."""
    klines = client.get_historical_klines(symbol, interval, limit=limit)
    df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                        'Close Time', 'Quote Asset Volume', 'Number of Trades', 
                                        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
    df['Close'] = df['Close'].astype(float) 
    return df[['Close']]


historical_data = get_historical_data('BTCUSDT')
print(historical_data.head())


historical_data['Prev Close'] = historical_data['Close'].shift(1)  
historical_data = historical_data.dropna()  


X = historical_data[['Prev Close']]
y = historical_data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(results.head())

mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, predictions, label='Predicted', color='orange')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.show()
