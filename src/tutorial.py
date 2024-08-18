import os
from datetime import datetime

import pandas as pd
from gym_trading_env.downloader import download
import gymnasium as gym
import gym_trading_env

# Comprobar si existe el fichero "./data/binance-BTCUSDT-1h.pkl", y si no, descargarlo
if not os.path.exists("./data/binance-BTCUSDT-1h.pkl"):
    download(
        exchange_names=["binance"],
        symbols=["BTC/USDT"],
        timeframe="1h",
        dir="data",
        since=datetime(year=2020, month=1, day=1)
    )

# Import your fresh data
df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
print(df.columns)

df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"] / df["close"]
df["feature_high"] = df["high"] / df["close"]
df["feature_low"] = df["low"] / df["close"]
df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()

df.dropna(inplace=True)
print(df.head())

env = gym.make(
    "TradingEnv",
    df=df,
    positions=[-1, 0, 1],
    trading_fees=0.01/100,
    borrow_interest_rate=0.0003/100,
)

done, truncated = False, False
observation, info = env.reset()
while not done and not truncated:
    position_index = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(position_index)
env.unwrapped.save_for_render(dir="render_logs")