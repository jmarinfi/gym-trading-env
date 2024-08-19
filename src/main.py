import os
from datetime import datetime

import gymnasium as gym
import pandas as pd
from gym_trading_env.downloader import download

from src.utils_agent.custom_reward import custom_reward

# Comprobar si existe el fichero "./data/binance-BTCUSDT-1h.pkl", y si no, descargarlo
path = os.path.abspath("data")
if not os.path.exists(path):
    download(
        exchange_names=["binance"],
        symbols=["BTC/USDT"],
        timeframe="1h",
        dir=path,
        since=datetime(year=2020, month=1, day=1)
    )

# Importar los datos
df = pd.read_pickle(os.path.join(path, "binance-BTCUSDT-1h.pkl"))
df.dropna(inplace=True)
print(df.columns)

# Creación del entorno de trading
env = gym.make(
    "TradingEnv",
    df=df,
    positions=[-1, 0, 1],
    trading_fees=0.075/100,
    borrow_interest_rate=0.0003/100,
    portfolio_initial_value=10000,
    reward_function=custom_reward,
    verbose=1,
)
env.unwrapped.add_metric('Accumulated reward', lambda history: sum(history['reward']))

# Entrenamiento del agente
done, truncated = False, False
observation, info = env.reset()

while not done and not truncated:
    position_index = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(position_index)

# env.unwrapped.save_for_render(dir="render_logs")
