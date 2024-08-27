import math
import random
from collections import namedtuple, deque
from datetime import datetime
from itertools import count

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils_data.manage_data import preprocess_data_env, download_data
from src.utils_env.custom_reward import simple_custom_reward


today = datetime.today()

download_data('../data', since=datetime(year=2020, month=1, day=1), until=datetime(year=2024, month=5, day=31), symbols=["BTC/USDC"])
download_data('../data_validation', since=datetime(year=2024, month=6, day=1), until=today, symbols=["BTC/USDC"])

def create_env(dataset_dir):
    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir=dataset_dir,
        preprocess=preprocess_data_env,
        positions=[-1, -0.5, 0, 0.5, 1, 2],
        trading_fees=0.075/100,
        borrow_interest_rate=0.0000016146,
        reward_function=simple_custom_reward,
        portfolio_initial_value=1000,
        verbose=1
    )
    env.unwrapped.add_metric('Accumulated reward', lambda history: sum(history['reward']))

    return env

# Dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


env_data = create_env('../data/*.pkl')
env_validation = create_env('../data_validation/*.pkl')


# Hiperparámetros
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env_data.action_space.n
state, info = env_data.reset()
n_observations = len(state)

print(f"n_observations: {n_observations}")
print(f"n_actions: {n_actions}")
print(f"state: {state}")
print(f"info: {info}")

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env_data.action_space.sample()]], device=device, dtype=torch.long)

episode_rewards = []

def plot_rewards(episode, show_result=False):
    plt.figure(2)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        # plt.clf()
        plt.title('Training... Episode: ' + str(episode))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # plt.pause(0.001)
    plt.savefig(f'episode_{episode}.png')
    plt.close()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

num_episodes = 1000

for i_episode in range(num_episodes):
    state, info = env_data.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env_data.step(action.item())
        reward = torch.tensor([reward], device=device)
        episode_reward += reward.item()
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_rewards.append(episode_reward)
            plot_rewards(episode=i_episode)
            break

    state, info = env_validation.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done, truncated = False, False
    portfolio_values = []
    benchmark_values = []
    initial_close = info['data_close']
    initial_portfolio = info['portfolio_valuation']

    while not done and not truncated:
        action = policy_net(state).max(1).indices.view(1, 1)
        observation, reward, done, truncated, info = env_validation.step(action.item())
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        state = next_state

        portfolio_values.append(info['portfolio_valuation'])
        current_close = info['data_close']
        benchmark_values.append(initial_portfolio * (current_close / initial_close))

    portfolio_values = np.array(portfolio_values)
    benchmark_values = np.array(benchmark_values)

    plt.plot(portfolio_values, label='Portfolio')
    plt.plot(benchmark_values, label='Benchmark')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'portfolio_vs_benchmark_{str(i_episode)}.png')
    plt.close()


print('Complete')
plot_rewards(episode=num_episodes ,show_result=True)
# plt.ioff()
# plt.show()



env_data.close()
env_validation.close()
