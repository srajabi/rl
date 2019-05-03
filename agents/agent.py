import numpy as np
import collections
import random
import time
from abc import ABC, abstractmethod
from agents.epsilon_schedule import epsilon_schedule
from agents.constants import MIN_EPSILON, DISCOUNT_RATE
from agents.model import save_model, load_latest_model
from agents.utils.plot import save_plot
from keras.utils.np_utils import to_categorical


class Agent(ABC):

    def __init__(self,
                 memlength,
                 env):
        self._assert_environment(env)

        self.memory = collections.deque(maxlen=memlength)
        self.env = env

        self.action_space = env.action_space.n

        self.model = self._build_model(self.action_space)

        self.scores = []
        self.ep_lengths = []
        self.history = []

    def _choose_action(self, state, total_eps, cur_ep):
        if random.uniform(0, 1) < epsilon_schedule(total_eps, cur_ep, MIN_EPSILON):
            return self.env.action_space.sample()
        else:
            return self._predict_action(state)

    def _predict_action(self, state):
        mask = np.ones(self.action_space).reshape(1, self.action_space)
        q_values = self.model.predict([np.expand_dims(state, axis=0), mask])
        return np.argmax(q_values[0])

    def play(self, n_episodes):
        for ep in range(0, n_episodes):
            prev_state = self.env.reset()
            prev_state = self._preprocess_state_initial(prev_state)

            ep_reward = 0
            ep_len = 0

            done = False
            while not done:
                action = self._predict_action(prev_state)

                self.env.render()
                state_raw, reward, done, info = self.env.step(action)

                state = self._preprocess_state(state_raw, prev_state)

                prev_state = state

                ep_reward += reward
                ep_len += 1

            self.env.render()

            self.scores.append(ep_reward)
            self.ep_lengths.append(ep_len)

    def train(self, n_episodes):
        try:
            for ep in range(0, n_episodes):
                start_time = time.time()

                prev_state = self.env.reset()
                prev_state = self._preprocess_state_initial(prev_state)

                self._reset()

                ep_reward = 0
                ep_len = 0

                done = False
                while not done:
                    action = self._choose_action(prev_state, n_episodes, ep)

                    state_raw, reward, done, info = self.env.step(action)

                    lost_life = False
                    if not done:
                        lost_life = self._stepped()

                    terminal = lost_life or done
                    state = self._preprocess_state(state_raw, prev_state)
                    self.memory.append([prev_state, action, reward, state, terminal])

                    prev_state = state

                    ep_reward += reward
                    ep_len += 1

                    self.train_model()

                self.scores.append(ep_reward)
                self.ep_lengths.append(ep_len)

                total_episode_time = time.time() - start_time

                self.print_stats(ep, total_episode_time)

                if ep % 1000 == 0:
                    self.checkpoint()

        except KeyboardInterrupt:
            self.checkpoint()

    def checkpoint(self):
        if len(self.history) == 0:
            return

        save_model(self.model, self._agent_name())

        sample_freq = max(len(self.history)//100, 1)
        save_plot('Loss',
                  'Frame',
                  'Loss',
                  self.history[::sample_freq],
                  range(0, len(self.history), sample_freq))

        sample_freq = max(len(self.ep_lengths)//100, 1)
        save_plot('Episode_Lengths',
                  'Episode',
                  'Ep Len',
                  self.ep_lengths[::sample_freq],
                  range(0, len(self.ep_lengths), sample_freq))

        window = 5
        running_scores = np.convolve(self.scores, np.ones((window,)) / window, mode='valid')

        sample_freq = max(len(running_scores)//100, 1)
        sampled_scores = running_scores[::sample_freq]

        lower_bound = window - 1
        upper_bound = len(running_scores) + (window - 1)

        save_plot('Running_Score',
                  'Episode',
                  'Avg Score Last 5 Games',
                  sampled_scores,
                  range(lower_bound, upper_bound, sample_freq))

    def print_stats(self, episode, total_ep_time):
        if len(self.history) <= 5:
            return

        last_scores = self.scores[-5:]
        last_losses = self.history[-5:]
        last_ep_len = self.ep_lengths[-5:]

        avg_score = sum(last_scores)/len(last_scores)
        avg_loss = sum(last_losses)/len(last_losses)
        avg_len = sum(last_ep_len)/len(last_ep_len)

        # TODO Graph
        print('ep: {0} len: {1:.2f} loss: {2:.5f} scoreavg: {3:.2f} score: {4} time: {5:.2f}'.format
              (episode,
               avg_len,
               avg_loss,
               avg_score,
               self.scores[-1],
               total_ep_time))

    def train_model(self):
        mem_len = len(self.memory)

        if mem_len < 32:
            return

        prev_states = np.empty((32,) + self._observation_shape())
        actions = np.empty((32, self.action_space))
        rewards = np.empty(32)
        next_states = np.empty((32,) + self._observation_shape())
        dones = np.empty(32, dtype=bool)

        rand_idxs = np.random.randint(0, mem_len - 1, size=32)

        for idx, mem_idx in enumerate(rand_idxs):
            cur_mem = self.memory[mem_idx]

            prev_states[idx] = cur_mem[0]
            actions[idx] = to_categorical(cur_mem[1],
                                          self.action_space)
            rewards[idx] = cur_mem[2]
            next_states[idx] = cur_mem[3]
            dones[idx] = cur_mem[4]

        estimated_nexts = self.model.predict([next_states, np.ones((32, self.action_space))])

        y = []
        for idx in range(0, 32):
            est_next = estimated_nexts[idx]
            est_q = np.max(est_next)

            action_oneh = actions[idx]
            action = np.argmax(action_oneh)

            done = dones[idx]
            reward = rewards[idx]

            if done:
                est_next[action] = reward
            else:
                est_next[action] = (
                        reward + DISCOUNT_RATE * est_q
                )

            y.append(est_next)

        y = np.asarray(y)

        history = self.model.fit([prev_states, actions], y, batch_size=32, epochs=1, verbose=0)

        self.history.append(history.history['loss'][0])

    def load_model(self):
        self.model = load_latest_model(self._agent_name())

    @abstractmethod
    def _assert_environment(self, env):
        pass

    @abstractmethod
    def _preprocess_state_initial(self, state):
        pass

    @abstractmethod
    def _preprocess_state(self, state, prev_state):
        pass

    @abstractmethod
    def _build_model(self, action_space):
        pass

    @abstractmethod
    def _agent_name(self):
        pass

    @abstractmethod
    def _observation_shape(self):
        pass

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _stepped(self):
        pass
