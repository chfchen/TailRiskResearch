import yfinance as yf
import numpy as np
from google.colab import files
# Define ticker symbol and date range
ticker_symbol = "SPY"
start_date = "1995-12-31"
end_date = "2026-01-01"  # Set to Jan 1, 2026 to include Dec 31, 2025

print(f"Fetching SPY data from {start_date} to 2025-12-31...")

# 1. Download market data
df = yf.download(
    ticker_symbol,
    start=start_date,
    end=end_date,
    auto_adjust=False
)
df.to_csv("SPY.csv")
files.download("SPY.csv")
