import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class PerformanceTracker:
    def __init__(self, window_size=100):
        self.rewards = []
        self.portfolio_values = []
        self.epsilon_values = []

    def update(self, reward, portfolio_value, epsilon):
        self.rewards.append(reward)
        self.portfolio_values.append(portfolio_value)
        self.epsilon_values.append(epsilon)


def plot_training_performance(tracker, iteration):
    fig, axs = plt.subplots(1, 3, figsize=(25, 5))

    # Plot rewards
    axs[0].plot(tracker.rewards)
    axs[0].set_title('Rewards per Episode. Iteration: ' + str(iteration))
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('Total Reward')

    # Plot portfolio values
    axs[1].plot(tracker.portfolio_values)
    axs[1].set_title('Portfolio Value. Iteration: ' + str(iteration))
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Value')

    # Plot epsilon values
    axs[2].plot(tracker.epsilon_values)
    axs[2].set_title('Epsilon Values. Iteration: ' + str(iteration))
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig(f'training_performance_{str(iteration)}.png')
    plt.close()


def plot_evaluation_performance(portfolio_values, benchmark_values, iteration):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Agent Portfolio')
    plt.plot(benchmark_values, label='Benchmark (Buy and Hold)')
    plt.title('Portfolio value vs Benchmark. Iteration: ' + str(iteration))
    plt.xlabel('Step')
    plt.ylabel('Portfolio value')
    plt.legend()
    plt.savefig(f'evaluation_performance_{str(iteration)}.png')
    plt.close()


def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def evaluate_agent_performance(portfolio_values, benchmark_values):
    agent_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]

    agent_sharpe = calculate_sharpe_ratio(agent_returns)
    benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns)

    max_drawdown_agent = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values)
    max_drawdown_benchmark = np.max(np.maximum.accumulate(benchmark_values) - benchmark_values) / np.max(benchmark_values)

    print(f"Agent Sharpe Ratio: {agent_sharpe:.2f}")
    print(f"Benchmark Sharpe Ratio: {benchmark_sharpe:.2f}")
    print(f"Agent Max Drawdown: {max_drawdown_agent:.2%}")
    print(f"Benchmark Max Drawdown: {max_drawdown_benchmark:.2%}")
    print(f"Total Return Agent: {(portfolio_values[-1] / portfolio_values[0] - 1):.2%}")
    print(f"Total Return Benchmark: {(benchmark_values[-1] / benchmark_values[0] - 1):.2%}")
