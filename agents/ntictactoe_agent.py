import gym
import gym_ntictactoe
from agents.model import build_dense_model
from keras.utils.np_utils import to_categorical
from agents.selfplay_agent import SelfPlayAgent
from gym.spaces.discrete import Discrete


class NTicTacToeAgent(SelfPlayAgent):

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
        return state

    def _preprocess_state(self, state, prev_state):
        return state

    def _build_model(self, action_space):
        return build_dense_model(self.observation_space, action_space)

    def _agent_name(self):
        return self.__class__.__name__

    def _observation_shape(self):
        return (self.observation_space,)


if __name__ == "__main__":
    env = gym.make('tictactoe-v0')

    agent = NTicTacToeAgent(env)
    agent.train(100000)

    agent.load_model()
    agent.play(10)
