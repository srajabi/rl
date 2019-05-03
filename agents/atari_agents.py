import gym
import numpy as np
from agents.model import build_larger_model
from agents.agent import Agent
from agents.constants import ATARI_INPUT_SHAPE
from gym.spaces.box import Box
from agents.img_processing.cv2_obs_processing import grayscale_crop
from agents.utils.profiling import profile_sort_and_print


class DeepQAgentAtari(Agent):

    def __init__(self, env):
        super().__init__(1000000, env)
        self.lives = 0


    def _assert_environment(self, env):
        assert isinstance(env.observation_space, Box),\
            "Error environment must have Box(210, 160, 3) action space."

    def _preprocess_state_initial(self, state):
        state = grayscale_crop(state)
        return np.dstack([state,
                          state,
                          state,
                          state])

    def _reset(self):
        self.lives = self.env.unwrapped.ale.lives()

    def _stepped(self):
        cur_lives = self.env.unwrapped.ale.lives()
        if self.lives > cur_lives:
            return True
        else:
            return False

    def _preprocess_state(self, state, prev_state):
        state = grayscale_crop(state)
        return np.append(state,
                         prev_state[:, :, :3],
                         axis=2)

    def _build_model(self, action_space):
        return build_larger_model(ATARI_INPUT_SHAPE, action_space)

    def _agent_name(self):
        return self.__class__.__name__

    def _observation_shape(self):
        return ATARI_INPUT_SHAPE


if __name__ == "__main__":
    profile_sort_and_print("env = gym.make('BreakoutDeterministic-v4')\n"
                           "agent = DeepQAgentAtari(env)\n"
                           "agent.train(10)")

    env = gym.make('BreakoutDeterministic-v4')

    agent = DeepQAgentAtari(env)
    agent.train(1000000)

    agent.load_model()
    agent.play(3)
