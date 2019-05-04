import gym
import gym_ntictactoe
import time

env = gym.make('tictactoe-v0')

done = False
env.reset()
while not done:
    env.render()
    str = input("Input Action: 'Player Move',  where Player is {-1, 1} and move is flattened index: \n")
    action = tuple(map(int, str.split(' ')))
    print(action)
    obv, reward, done, info = env.step(action)

    print(obv)
    print(reward)
    print(done)
    print(info)
