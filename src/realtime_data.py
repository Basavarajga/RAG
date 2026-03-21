import yfinance as yf

def get_stock_price(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")

        if data.empty:
            return None

        price = data["Close"].iloc[-1]
        return float(price)

    except Exception:
        return None
      
