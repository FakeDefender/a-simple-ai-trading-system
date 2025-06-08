import ccxt

exchange = ccxt.binance()
markets = exchange.load_markets()
symbols = list(markets.keys())
print(symbols)