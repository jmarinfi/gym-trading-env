from datetime import datetime
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
import pandas as pd
from gym_trading_env.downloader import download

from src.utils_agent.custom_reward import custom_reward
from src.utils_agent.ddqn_agent import DDQNAgent


# download(
#     exchange_names=["binance"],
#     symbols=["BTC/USDT"],
#     timeframe="1h",
#     dir='data',
#     since=datetime(year=2020, month=6, day=1)
# )

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def preprocess(df: pd.DataFrame):
    df['feature_ma7'] = df['close'].rolling(window=7).mean() / df['close'] - 1
    df['feature_ma30'] = df['close'].rolling(window=30).mean() / df['close'] - 1
    df['feature_rsi'] = calculate_rsi(df['close'], window=14)
    df['feature_close'] = df['close'].pct_change()
    df['feature_volume'] = df['volume'] / df['volume'].rolling(window=7 * 24).max()
    df.dropna(inplace=True)

    return df


# Creación del entorno de trading
env = gym.make(
    "MultiDatasetTradingEnv",
    dataset_dir='data/*.pkl',
    preprocess=preprocess,
    positions=[-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5],
    trading_fees=0.075 / 100,
    borrow_interest_rate=0.0003 / 100,
    reward_function=custom_reward,
    windows=30,
    verbose=1,
    max_episode_duration=168
)
env.unwrapped.add_metric('Accumulated reward', lambda history: sum(history['reward']))

print(f"Observation space shape: {env.observation_space.shape}")
print(f"Columnas df: {env.unwrapped.df.columns.tolist()}")
print(f"Action size: {env.action_space.n}")

NUM_AGENT_FEATURES = 2

agent = DDQNAgent(
    sequence_length=env.observation_space.shape[0],
    batch_size=32,
    num_actions=env.action_space.n,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    learning_rate=0.001,
    num_features=env.observation_space.shape[1],
    gamma=0.95,
    num_inner_neurons=128,
    market_feature_size=env.observation_space.shape[1] - NUM_AGENT_FEATURES,
    agent_feature_size=NUM_AGENT_FEATURES,
    use_separate_networks=True
)

# Parámetros de entrenamiento
num_episodes = 100
update_target_frequency = 10
save_frequency = 10

# Listas para almacenar métricas
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.expand_dims(state, axis=0)
    total_reward = 0
    initial_portfolio_value = env.unwrapped.portfolio_initial_value
    done, truncated = False, False

    while not done and not truncated:
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(agent.memory) > agent.batch_size:
            agent.replay()

    # Actualizar el modelo objetivo
    if episode % update_target_frequency == 0:
        agent.update_target_model()

    # Guardar el modelo
    if episode % save_frequency == 0:
        agent.save(f"ddqn_model_episode_{episode}")

    episode_rewards.append(total_reward)

    print(f"Info episode {episode}: {info}")
    print(f"Episode: {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}")
    print(episode_rewards)

    # Reducir epsilon
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

# Graficar resultados
plt.figure(figsize=(12, 4))
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.tight_layout()
plt.savefig('training_results.png')
plt.close()

# Guardar modelo final
agent.save("ddqn_final_model.keras")

# env.unwrapped.save_for_render(dir="render_logs")

env.close()
