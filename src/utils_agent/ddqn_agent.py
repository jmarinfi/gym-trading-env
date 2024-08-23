from collections import deque
import random

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam


class DDQNAgent:
    def __init__(self, sequence_length, num_features, num_inner_neurons, batch_size, num_actions, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, market_feature_size, agent_feature_size, use_separate_networks):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_inner_neurons = num_inner_neurons
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.market_feature_size = market_feature_size
        self.agent_feature_size = agent_feature_size
        self.use_separate_networks = use_separate_networks

    def build_model(self):
        model = Sequential([
            Input(shape=(self.sequence_length, self.num_features)),
            LSTM(self.num_inner_neurons, return_sequences=True),
            LSTM(self.num_inner_neurons),
            Dense(self.num_inner_neurons, activation='relu'),
            Dense(self.num_actions, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Muestreo aleatorio de la memoria
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([experience[0] for experience in minibatch]) # States shape: (32, 1, 30, 7)
        actions = np.array([experience[1] for experience in minibatch]) # Actions shape: (32,)
        rewards = np.array([experience[2] for experience in minibatch]) # Rewards shape: (32,)
        next_states = np.array([experience[3] for experience in minibatch]) # Next states shape: (32, 1, 30, 7)
        dones = np.array([experience[4] for experience in minibatch]) # Dones shape: (32,)

        states = np.squeeze(states, axis=1) # States shape: (32, 30, 7)
        next_states = np.squeeze(next_states, axis=1) # Next states shape: (32, 30, 7)

        # Predicciones del modelo principal
        target = self.model.predict(states, verbose=0) # Target shape: (32, 1)

        # Predicciones del modelo objetivo
        target_next = self.target_model.predict(next_states, verbose=0) # Target next shape: (32, 1)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                # Implementación de Double DQN
                a = np.argmax(self.model.predict(next_states[i:i + 1], verbose=0)[0])
                target[i][actions[i]] = rewards[i] + self.gamma * target_next[i][a]

        self.model.fit(states, target, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save(f"{name}.keras")
