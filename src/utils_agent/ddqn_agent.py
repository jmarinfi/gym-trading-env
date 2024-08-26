from collections import deque
import random

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.models import load_model


class DDQNAgent:
    def __init__(self, sequence_length, num_features, num_inner_neurons, batch_size, num_actions, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, is_eval=False):
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
        self.is_eval = is_eval

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
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        # Muestreo aleatorio de la memoria
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([experience[0] for experience in minibatch]) # States shape: (batch_size, num_envs, window, num_features)
        actions = np.array([experience[1] for experience in minibatch]) # Actions shape: (batch_size, num_envs)
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        if len(states.shape) == 4:
            batch_size, num_envs, windows, num_features = states.shape
        else:
            batch_size, windows, num_features = states.shape
            num_envs = 1
        states = np.reshape(states, (batch_size * num_envs, windows, num_features))
        next_states = np.reshape(next_states, (batch_size * num_envs, windows, num_features))
        if len(actions.shape) == 2:
            actions = np.reshape(actions, (batch_size * num_envs))
        if len(rewards.shape) == 2:
            rewards = np.reshape(rewards, (batch_size * num_envs))
        if len(dones.shape) == 2:
            dones = np.reshape(dones, (batch_size * num_envs))

        # Predicciones del modelo principal
        target = self.model.predict(states, verbose=0) # Target shape: (32, 1)

        # Predicciones del modelo objetivo
        target_next = self.target_model.predict(next_states, verbose=0) # Target next shape: (32, 1)

        for i in range(batch_size * num_envs):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                # Implementación de Double DQN
                a = np.argmax(self.model.predict(next_states[i:i + 1], verbose=0)[0])
                target[i][actions[i]] = rewards[i] + self.gamma * target_next[i][a]

        self.model.fit(states, target, epochs=1, verbose=0)

    def load(self, name):
        # self.model.load_weights(name)
        self.model = load_model(name)

    def save(self, name):
        self.model.save(f"{name}.keras")
