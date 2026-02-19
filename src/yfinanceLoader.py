import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Download 1 year of Apple stock data
data = yf.download("AAPL", period="5y")

print("\nFull DataFrame:\n")
print(data.head())

# Extract Close prices
y = data["Close"].astype("float32").to_numpy()

print("\nClose Prices (NumPy array):\n")
print(y)
print("\nShape:", y.shape)

# -----------------------
# Plot full year
# -----------------------
plt.figure()
plt.plot(data.index, y)
plt.title("AAPL Close Price - 1 Year")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------
# Plot zoom (first 100 days)
# -----------------------
plt.figure()
plt.plot(data.index[:100], y[:100])
plt.title("AAPL Close Price - First 100 Days")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()