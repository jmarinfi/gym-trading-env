import numpy as np
import pandas as pd
from gym_trading_env.downloader import download
from sklearn.preprocessing import robust_scale


def download_data(dir, since, until, exchange_names = ["binance"], symbols = ["BTC/USDC"], timeframe = "1h"):
    download(
        exchange_names=exchange_names,
        symbols=symbols,
        timeframe=timeframe,
        dir=dir,
        since=since,
        until=until
    )

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def preprocess_data_env(df: pd.DataFrame):
    df['feature_ma7'] = robust_scale(df['close'].rolling(window=7).mean())
    df['feature_ma30'] = robust_scale(df['close'].rolling(window=30).mean())
    df['feature_rsi'] = robust_scale(calculate_rsi(df['close'], window=14))
    df["feature_close"] = robust_scale(df["close"].pct_change().replace([np.inf, -np.inf], np.nan))
    df["feature_open"] = robust_scale(df["open"] / df["close"])
    df["feature_high"] = robust_scale(df["high"] / df["close"])
    df["feature_low"] = robust_scale(df["low"] / df["close"])
    df["feature_volume"] = robust_scale(df["volume"].pct_change().replace([np.inf, -np.inf], np.nan))
    df.dropna(inplace=True)
    return df
