import os.path
import time
from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.utils_agent.ddqn_agent import DDQNAgent
from src.utils_data.exchange_connector_binance import BinanceConnector, IntervalLetter
from src.utils_data.manage_data import download_data, preprocess_data_env
from src.utils_env.custom_reward import simple_custom_reward

load_dotenv()
api_key = os.getenv('API_KEY_BINANCE')
api_secret = os.getenv('SECRET_KEY_BINANCE')
binance_connector = BinanceConnector(api_key, api_secret)

model_path = os.path.abspath('ddqn_model_iteration_0.keras')
df_path = os.path.abspath('data_live/binance-BTCUSDC-5m.pkl')
df_training_path = os.path.abspath('data_live/binance-BTCUSDC-1h.pkl')

initial_portfolio_value = 1000
display_actions = [
    'Short sell', 'Short sell half', 'Hold', 'Buy', 'Buy double'
]

# Instanciar el agente y cargar el modelo
agent = DDQNAgent(
    sequence_length=30,
    batch_size=32,
    num_actions=5,
    epsilon=0.5, # 50% de exploración
    epsilon_min=0.01,
    epsilon_decay=1, # No decaer el epsilon
    learning_rate=0.001,
    num_features=10,
    gamma=0.99,
    num_inner_neurons=64
)
agent.load(model_path)

while True:
    agent.is_eval = True

    # Obtener los últimos datos del exchange
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=60)
    download_data('data_live', since=start_time, until=end_time, timeframe='5m')

    df = pd.read_pickle(df_path)
    first_timestamp = df.index[0]
    offset = pd.Timedelta(minutes=first_timestamp.minute, seconds=first_timestamp.second, microseconds=first_timestamp.microsecond)

    # Resamplear los datos a 1h
    df = df.resample('1h', offset=offset).agg({
        'date_close': lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
        'open': lambda x: x.iloc[0] if len(x) > 0 else np.nan,
        'high': lambda x: max(x) if len(x) > 0 else np.nan,
        'low': lambda x: min(x) if len(x) > 0 else np.nan,
        'close': lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
        'volume': lambda x: sum(x) if len(x) > 0 else np.nan
    })

    # Preprocesar los datos
    df = preprocess_data_env(df)[-30:]

    # Crear el entorno
    env = gym.make(
        'TradingEnv',
        df=df,
        positions=[-1, -0.5, 0, 1, 2],
        reward_function=simple_custom_reward,
        windows=30,
        trading_fees=0.01 / 100,
        borrow_interest_rate=0.0003 / 100,
        portfolio_initial_value=initial_portfolio_value,
        initial_position=0,
        verbose=1
    )

    # Obtener la acción del agente
    state, info = env.reset()
    action = agent.act(np.expand_dims(state, axis=0))
    print(f'Action: {display_actions[action]}')
    env.close()

    # Entrenar el modelo con los datos de la última semana
    start_time = end_time - timedelta(days=7)
    klines = binance_connector.get_klines('BTCUSDC', 1, IntervalLetter.HOUR, limit=24*7)
    df_training = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df_training['timestamp'] = pd.to_datetime(df_training['timestamp'], unit='ms')
    df_training.set_index('timestamp', inplace=True)
    df_training = df_training[['open', 'high', 'low', 'close', 'volume']]
    df_training[['open', 'high', 'low', 'close', 'volume']] = df_training[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df_training = preprocess_data_env(df_training)

    training_env = gym.make(
        'TradingEnv',
        df=df_training,
        positions=[-1, -0.5, 0, 1, 2],
        reward_function=simple_custom_reward,
        windows=30,
        trading_fees=0.01 / 100,
        borrow_interest_rate=0.0003 / 100,
        portfolio_initial_value=1000,
        initial_position=0,
        verbose=1
    )

    agent.is_eval = False

    state, info = training_env.reset()
    done, truncated = False, False

    while not done and not truncated:
        action = agent.act(np.expand_dims(state, axis=0))
        next_state, reward, done, truncated, info = training_env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

    agent.replay()
    agent.update_target_model()

    training_env.close()

    time.sleep(3600) # Esperar una hora