import numpy as np

from src.utils_env.performance_evaluation import PerformanceTracker


def train_agent(num_steps, update_target_frequency, decay_frequency, agent, envs):
    states, info = envs.reset()
    agent.is_eval = False
    tracker = PerformanceTracker()
    episode_reward = 0

    for step in range(num_steps):
        print(f'____Training step {step + 1}/{num_steps}____')
        actions = np.array([agent.act(np.expand_dims(state, axis=0)) for state in states])
        next_states, rewards, dones, truncateds, info = envs.step(actions)

        agent.remember(states, actions, rewards, next_states, dones)
        episode_reward += np.mean(rewards)

        if step > 0 and step % agent.batch_size == 0:
            print('__Replay__')
            agent.replay()

        if step > 0 and step % update_target_frequency == 0:
            print('__Update target model__')
            agent.update_target_model()

        if step > 0 and step % decay_frequency == 0:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        current_value = np.mean(info['portfolio_valuation'])
        tracker.update(episode_reward, current_value, agent.epsilon)
        episode_reward = 0

        states = next_states

    return tracker

def evaluate_agent(envs, agent):
    states, info = envs.reset()
    agent.is_eval = True
    check = np.full(envs.num_envs, False)

    portfolio_values = []
    benchmark_values = []
    initial_close = np.mean(info['data_close'])
    initial_portfolio = np.mean(info['portfolio_valuation'])

    while not np.all(check):
        actions = np.array([agent.act(np.expand_dims(state, axis=0)) for state in states])
        next_states, rewards, dones, truncateds, info = envs.step(actions)

        check += dones + truncateds
        if np.all(check):
            break

        portfolio_values.append(np.mean(info['portfolio_valuation']))
        current_close = np.mean(info['data_close'])
        benchmark_values.append(initial_portfolio * current_close / initial_close)

        states = next_states

    portfolio_values = np.array(portfolio_values)
    benchmark_values = np.array(benchmark_values)

    return portfolio_values, benchmark_values
