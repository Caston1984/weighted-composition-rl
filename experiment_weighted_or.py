
"""
Purple circle vs beige square as a function of weights
"""
import gym
import torch
import time
from gym.wrappers.monitor import Monitor
from random import random
from dqn import ComposedDQN, FloatTensor, get_action
from trainer import load
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaxLength(gym.Wrapper):
    def __init__(self, env, length):
        gym.Wrapper.__init__(self, env)
        self.max_length = length
        self.steps = 0

    def reset(self):
        self.steps = 0
        return self.env.reset()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.steps += 1
        if self.steps == self.max_length:
            done = True
        return ob, reward, done, info


#if __name__ == '__main__':
    def collect(self, max_iterations, max_episodes, max_trajectory, weight):
        self.max_iterations = max_iterations
        self.max_episodes = max_episodes
        self.max_trajectory = max_trajectory
        self.weight = weight

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        task = MaxLength(WarpFrame(CollectEnv(goal_condition=lambda x: (x.colour == 'beige' and x.shape == 'square')
                                                                   or (x.colour == 'purple' and x.shape == 'circle'))),
                     max_trajectory)
        env = Monitor(task, './experiment_weighted_or/', video_callable=False, force=True)

        dqn_purple_circle = load('./models/purple_circle/model.dqn', task)  # entropy regularised functions
        dqn_beige_crate = load('./models/beige_crate/model.dqn', task)  # entropy regularised functions
        dqn_purple_circle.to(device) #made these models to use GPU instead of CPU
        dqn_beige_crate.to(device) #made these models to use GPU instead of CPU
        #weights = np.arange(0, 150, 50)
        print(torch.cuda.is_available())
        #tally = {i: [] for i in range(len(weights))}
        tally = []

        for iter in range(max_iterations):
        #for i, weight in enumerate(weights):
            collected_count = [0, 0]
            #weight = 1
            dqn_composed = ComposedDQN([dqn_beige_crate, dqn_purple_circle], [weight, 1 - weight])

            for episode in range(max_episodes):
                if episode % 1000 == 0:
                    print(episode)
                obs = env.reset()

                for _ in range(max_trajectory):
                    obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
                    # action = dqn_composed(Variable(obs, volatile=True)).data.max(1)[1].view(1, 1)[0][0]
                    action = get_action(dqn_composed, obs)
                    obs, reward, done, info = env.step(action)
                    if done:
                        collected = info['collected']
                        if len([c for c in collected if c.colour == 'beige' and c.shape == 'square']) > 0:
                            collected_count[0] += 1
                        elif len([c for c in collected if c.colour == 'purple' and c.shape == 'circle']) > 0:
                            collected_count[1] += 1
                        else:
                            print("Missed")
                        break
            #tally[i].append(collected_count)
            tally.append(collected_count)

        mean_collected = np.round(np.array(tally).mean(axis = 0))

            #print('Weight = {}'.format(weight))
            #print(tally[i])
        return mean_collected

    def reward(self, max_trajectory, weight):
            #self.max_iterations = max_iterations
            #self.max_episodes = max_episodes
            self.max_trajectory = max_trajectory
            self.weight = weight

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            task = MaxLength(WarpFrame(CollectEnv(goal_condition=lambda x: (x.colour == 'beige' and x.shape == 'square')
                                                                           or (
                                                                                       x.colour == 'purple' and x.shape == 'circle'))),
                             max_trajectory)
            env = Monitor(task, './experiment_weighted_or/', video_callable=False, force=True)

            dqn_purple_circle = load('./models/purple_circle/model.dqn', task)  # entropy regularised functions
            dqn_beige_crate = load('./models/beige_crate/model.dqn', task)  # entropy regularised functions
            dqn_purple_circle.to(device)  # made these models to use GPU instead of CPU
            dqn_beige_crate.to(device)  # made these models to use GPU instead of CPU

            dqn_composed = ComposedDQN([dqn_beige_crate, dqn_purple_circle], [weight, 1 - weight])

            obs = env.reset()
            i = 0
            for _ in range(max_trajectory):
                #env.render() #works just turned it off because it delays the program
                #time.sleep(.1)
                obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
                # action = dqn_composed(Variable(obs, volatile=True)).data.max(1)[1].view(1, 1)[0][0]
                action = get_action(dqn_composed, obs)
                obs, reward, done, info = env.step(action)
                i+=1

                if done:
                    collected = info['collected']
                    if len([c for c in collected if c.colour == 'beige' and c.shape == 'square']) > 0:
                        return -1.0 * i + np.random.normal(0, 1)
                        #return i * -1.0 + 0.8, 'BS'
                    elif len([c for c in collected if c.colour == 'purple' and c.shape == 'circle']) > 0:
                        return -1.0 * i + np.random.random_sample()
                        #return i * -1.0 + 0.2, 'PC'
                    else:
                        return -1.0 * i + -1.0 + -1.0
                        #return i * -1.0 + -1.0, 'None'




        #print(tally)


    #with open('tally.json', 'w') as fp:
        #json.dump(tally, fp)