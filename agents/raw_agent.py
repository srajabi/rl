import gym
from agents.model import build_dense_model
from keras.utils.np_utils import to_categorical
from agents.agent import Agent
from gym.spaces.discrete import Discrete


class DeepQAgentNonVisual(Agent):

    def __init__(self, env):
        self.observation_space = env.observation_space.n
        super().__init__(10000, env)

    def _assert_environment(self, env):
        assert isinstance(env.observation_space, Discrete),\
            "Error environment must have discrete observation space."

    def _reset(self):
        pass

    def _stepped(self):
        pass

    def _preprocess_state_initial(self, state):
        return self._preprocess_state(state, None)

    def _preprocess_state(self, state, prev_state):
        return to_categorical(state, self.observation_space)

    def _build_model(self, action_space):
        return build_dense_model(self.observation_space, action_space)

    def _agent_name(self):
        return self.__class__.__name__

    def _observation_shape(self):
        return (self.observation_space,)


if __name__ == "__main__":
    env = gym.make('FrozenLake-v0', is_slippery=False)

    agent = DeepQAgentNonVisual(env)
    agent.train(10000)

    agent.load_model()
    agent.play(3)
