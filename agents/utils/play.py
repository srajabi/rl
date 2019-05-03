import gym
import gym_maze
import time
from gym.utils.play import play

env = gym.make('maze-v0')

keys_to_action = {
            (ord('w'),):0,
            (ord('s'),):1,
            (ord('d'),):2,
            (ord('a'),):3
        }

#play(env, keys_to_action=keys_to_action)


print(env.observation_space.low)
print(env.observation_space.high)

env = gym.make('FrozenLake8x8-v0')

print(env.observation_space.n)


done = False

env.reset()
while not done:
    #time.sleep(1)
    env.render()
    #action = env.action_space.sample()

    action = int(input('action'))
    print(action)
    obv, reward, done, info = env.step(action)

    print(obv)
    print(reward)
    print(done)
    print(info)
