from datetime import datetime
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from src.utils_env.custom_reward import simple_custom_reward
from src.utils_agent.ddqn_agent import DDQNAgent
from src.utils_data.manage_data import preprocess_data_env, download_data
from src.utils_env.performance_evaluation import plot_training_performance, plot_evaluation_performance, \
    evaluate_agent_performance
from src.utils_env.train_agent import train_agent, evaluate_agent

today = datetime.today()

download_data('data', since=datetime(year=2019, month=1, day=1), until=datetime(year=2023, month=5, day=31))
download_data('validation', since=datetime(year=2023, month=6, day=1), until=today)

def create_env(dataset_dir):
    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir=dataset_dir,
        preprocess=preprocess_data_env,
        positions=[-1, -0.5, 0, 1, 2],
        trading_fees=0.01 / 100,
        borrow_interest_rate=0.0003 / 100,
        reward_function=simple_custom_reward,
        portfolio_initial_value=1000,
        windows=30,
        verbose=1
    )

    env.unwrapped.add_metric('Accumulated reward', lambda history: sum(history['reward']))

    return env

envs_data = gym.vector.SyncVectorEnv([lambda: create_env('data/*.pkl') for _ in range(4)])
envs_validation = gym.vector.SyncVectorEnv([lambda: create_env('validation/*.pkl') for _ in range(4)])

agent = DDQNAgent(
    sequence_length=30,
    batch_size=32,
    num_actions=5,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    learning_rate=0.001,
    num_features=10,
    gamma=0.99,
    num_inner_neurons=64
)

# Parámetros de entrenamiento
num_iterations = 10
steps_per_iteration = 10000
update_target_frequency = 1000
decay_frequency = 1000

train_rewards = []
val_portfolio_values = []
val_benchmark_values = []

for iteration in range(num_iterations):
    print(f'Iteration {iteration}/{num_iterations}')

    tracker_train = train_agent(steps_per_iteration, update_target_frequency, decay_frequency, agent, envs_data)
    plot_training_performance(tracker_train, iteration=iteration)
    train_rewards.append(np.mean(tracker_train.rewards[-steps_per_iteration:]))

    portfolio_values, benchmark_values = evaluate_agent(envs_validation, agent)
    plot_evaluation_performance(portfolio_values, benchmark_values, iteration=iteration)
    evaluate_agent_performance(portfolio_values, benchmark_values)
    val_portfolio_values.append(portfolio_values[-1])
    val_benchmark_values.append(benchmark_values[-1])

    agent.save(f"ddqn_model_iteration_{iteration}")

    print(f'Training Reward: {train_rewards[-1]}')
    print(f'Validation Portfolio Value: {val_portfolio_values[-1]}')
    print(f'Validation Benchmark Value: {val_benchmark_values[-1]}')
    print('----------------------------------------------------------')

# Plot overall training and validation performance
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(train_rewards)
plt.title('Training Rewards per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Average Reward')

plt.subplot(2, 1, 2)
plt.plot(val_portfolio_values, label='Agent Portfolio')
plt.plot(val_benchmark_values, label='Benchmark')
plt.title('Validation Performance per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Final Portfolio Value')
plt.legend()

plt.tight_layout()
plt.savefig('overall_performance.png')
plt.close()

envs_data.close()
envs_validation.close()
